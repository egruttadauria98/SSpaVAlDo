"""
Training script based on Pyannote Speaker Diarization task.
"""

import os
import sys
import json
import argparse
from pprint import pprint
from config import common_parameters, conf

import torch
from types import MethodType
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping
from schedulers import BatchSizeScheduler, WeightDecayScheduler, compute_total_number_steps
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from task_enhanced import SpeakerDiarizationEnhanced
from speaker_diarization.data.pyannote_data_protocol import PyannoteDataProtocol
from speaker_diarization.utils.ssep_utils import load_ss_model
from speaker_diarization.utils.vad_utils import load_vad_model
from speaker_diarization.models.pyannote_ss_vad_model import SS_VAD_model
from speaker_diarization.utils.print_utils import print_args_decorator
from speaker_diarization.utils.train_utils import merge_dicts

from score_individual_segments import run_scoring

parser = argparse.ArgumentParser()
parser.add_argument("--conf_id", required=True,
                    help="Conf tag, used to get the right config")
parser.add_argument("--debug", type=bool, default=False,
                    help="If true save to specific directory")

@print_args_decorator
def main(conf):
  """Main function to run the training.

  Parameters
  ----------
  conf : dict
    Configuration dictionary.
  """

  conf_id = conf["conf_id"]
  if conf["debug"]:
    exp_dir = os.path.join(conf["exp_dir"], "DEBUG")
  else:
    exp_dir = os.path.join(conf["exp_dir"], f"conf_id_{conf_id}", conf["job_id"])
  
  conf['exp_dir_final'] = exp_dir

  # Load Pyannote data protocol
  pyannote_data = PyannoteDataProtocol(db_name=conf["data"]["db_name"])
  pyannote_data.load_protocol(protocol_name=conf["data"]["protocol_name"])
  assert pyannote_data.sample_rate == conf["model"]["sample_rate"], "Sample rate of model and data do not match"

  # Load SS model
  ss_model = load_ss_model(ss_name=conf["model"]["ss_model_name"],
                           ss_ckpt=conf["model"]["ss_model_ckpt"])

  # Load VAD model
  vad_model = load_vad_model(vad_name=conf["model"]["vad_model_name"],
                             vad_ckpt=conf["model"]["vad_model_ckpt"])

  # Assemble SS-VAD model
  model = SS_VAD_model(ss_model, 
                       vad_model, 
                       ss_freeze=conf["model"]["ss_freeze"], 
                       vad_max_method=conf["model"]["vad_max_method"])

  pprint(model)

  # Set duration of input segments for training
  model.setup_specifications(duration=conf["data"]["segment"])
  print(f"Working with segments of {model.specifications.duration}s")

  # Default diarization loss is NLL
  # If additional VAD loss is specified, the sum is used
  task = SpeakerDiarizationEnhanced(
      protocol=pyannote_data.protocol,
      duration=model.specifications.duration, 
      batch_size=conf["training"]["batch_size"],
      num_workers=conf["training"]["num_workers"],
      vad_loss=conf["loss"]["vad_loss"],
      max_speakers_per_chunk=conf["data"]["max_speakers_per_chunk"])

  model.task = task
  model.setup(stage="fit")

  optimizer = torch.optim.AdamW(model.parameters(), **conf["optim"])

  if conf["training"]["batch_scheduler"]:
    total_steps = compute_total_number_steps(task.train__len__(),
                                             conf["training"]["batch_size"],
                                             conf["training"]["max_epochs"],
                                             conf["training"]["devices"] *
                                             conf["training"]["num_nodes"],
                                             conf["training"]["batch_sizes"],
                                             conf["training"]["epochs"])
  else:
    if conf["training"]["wd_scheduler"]:
      total_steps = conf["training"]["max_epochs"] * (task.train__len__() // (
          conf["training"]["batch_size"]
          * conf["training"]["devices"]
          * conf["training"]["num_nodes"]))
    else:
      # Not used
      total_steps = None

  # LR scheduler
  if conf["training"]["lr_scheduler"]:
    if conf["training"]["lr_scheduler"] == "OneCycleLR":
      # Only for OneCycleLR, add `total_steps` in configs to pass
      conf["training"]["lr_scheduler_args"]["total_steps"] = total_steps

    # Define the LR scheduler, which can be OneCycleLR, ReduceLROnPlateau, etc.
    scheduler = getattr(torch.optim.lr_scheduler,
                        conf["training"]["lr_scheduler"])(optimizer, 
                                                          **conf["training"]["lr_scheduler_args"])
  else:
    scheduler = None

  # Monitor diarization error rate on the validation set
  # Use val loss to keep the best checkpoint and stop early
  monitor, direction = task.val_monitor
  print(f"Monitoring {monitor} with {direction} direction")

  # Set up Optimizer and LR scheduler
  def configure_optimizers(self):
      return {"optimizer": optimizer,
              "lr_scheduler": scheduler,
              "monitor": monitor}
  model.configure_optimizers = MethodType(configure_optimizers, model)

  # Define callbacks
  callbacks = [RichProgressBar()]

  checkpoint = ModelCheckpoint(
    os.path.join(exp_dir, "checkpoints/"),
    monitor=monitor, 
    mode=direction,
    save_top_k=5,
    every_n_epochs=1,
    save_last=True,
    save_weights_only=False,
    filename="{epoch}",
    verbose=True,
  )

  callbacks.append(checkpoint)

  if conf["training"]["early_stop"]:
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=direction,
        min_delta=conf["training"]["early_stop_min_delta"],
        patience=conf["training"]["early_stop_patience"],
        strict=True,
        verbose=True,
    )
    callbacks.append(early_stopping)

  # Control batch size
  if conf["training"]["batch_scheduler"]:
    callbacks.append(BatchSizeScheduler(
                        conf["training"]["batch_sizes"], 
                        conf["training"]["epochs"]
                        )
                    )
  # Control weight decay
  if conf["training"]["wd_scheduler"]:
    conf["training"]["wd_scheduler_args"]["total_steps"] = total_steps
    callbacks.append(WeightDecayScheduler(**conf["training"]["wd_scheduler_args"]))

  # Set up accelerator
  accelerator = "gpu" if torch.cuda.is_available() else "cpu"
  print(f"Accelerator: {accelerator}")

  # Train setup
  trainer = Trainer(accelerator=accelerator, 
                    callbacks=callbacks, 
                    max_epochs=conf["training"]["max_epochs"],
                    gradient_clip_val=conf["training"]["gradient_clip_val"],
                    num_nodes=conf["training"]["num_nodes"],
                    devices=conf["training"]["devices"],
                    # strategy="ddp_find_unused_parameters_true",
                    # limit_train_batches=0.01,  # Useful for fast experiment
                    # limit_val_batches=.01,  # Useful for fast experiment
                    reload_dataloaders_every_n_epochs=1,
                    log_every_n_steps=1,
                    default_root_dir=exp_dir,
                    logger=TensorBoardLogger(exp_dir, 
                                             log_graph=True,
                                             default_hp_metric=False))
  
  # Run training
  trainer.fit(model)
  print(f"\nBest model path: {checkpoint.best_model_path}\n")

  conf["best_model_path"] = checkpoint.best_model_path
  with open(os.path.join(exp_dir, "conf_end_train.json"), "w") as f:
    json.dump(conf, f, indent=0)

  # Record top 5 systems
  best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
  with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
    json.dump(best_k, f, indent=0)

  # Run scoring on individual segments
  run_scoring(best_model_path=checkpoint.best_model_path,
              ss_model=ss_model,
              vad_model=vad_model,
              exp_dir=exp_dir,
              conf=conf,
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
              )


if __name__ == "__main__":
  args = parser.parse_args()
  args = vars(args)

  conf = merge_dicts(common_parameters, conf[args["conf_id"]])
  conf = merge_dicts(conf, args)

  conf["data"]["db_name"] = conf["data"]["protocol_name"].split(".")[0]
  if conf["data"]["max_speakers_per_chunk"] == "n_src":
    conf["data"]["max_speakers_per_chunk"] = conf["model"]["n_src"]

  assert conf["model"]["ss_model_name"] == "ConvTasNet"
  if conf["model"]["ss_model_ckpt"] is None:
    sys.path.append("../")
    from ssep_models_dict import ssep_models_dict
    conf["model"]["ss_model_ckpt"] = ssep_models_dict\
                                      [conf["model"]["sample_rate"]]\
                                      [conf["model"]["n_src"]]

  if conf["model"]["vad_model_ckpt"] is None:
    sys.path.append("../")
    from vad_models_dict import vad_models_dict
    conf["model"]["vad_model_ckpt"] = vad_models_dict\
                                        [conf["model"]["sample_rate"]]\
                                        [conf["model"]["vad_model_name"]]

  pprint(conf)
  main(conf)

