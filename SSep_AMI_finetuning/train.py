"""
Training script to finetune pretrained SSep models on real data segments from AMI.
"""

import os
import sys
import json
import yaml
import argparse
from pprint import pprint
from config import common_parameters, conf

from asteroid.models import ConvTasNet
from asteroid.engine.optimizers import make_optimizer
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger

from merge_dicts import merge_dicts
from system_with_logging import System
from ami_segments_dataloader import AMI_segments

parser = argparse.ArgumentParser()
parser.add_argument("--conf_id", required=True,
                    help="Conf tag, used to get the right config")
parser.add_argument("--debug", type=bool, default=False,
                    help="If true save to specific directory")

def load_ss_model(ss_name, ss_ckpt):
  assert ss_name in ["ConvTasNet", "DPRNN", "SuDoRMRF"]
  if ss_name == "ConvTasNet":
    from asteroid.models import ConvTasNet
    try:
        ss_model = ConvTasNet.from_pretrained(ss_ckpt)
    except ValueError:
        from speaker_diarization.utils.torch_utils import upload_state_dict
        ss_model = ConvTasNet.from_pretrained(upload_state_dict(ss_ckpt))
  elif ss_name == "DPRNN":
    from asteroid.models import DPRNNTasNet
    try:
        ss_model = DPRNNTasNet.from_pretrained(ss_ckpt)
    except ValueError:
        from speaker_diarization.utils.torch_utils import upload_state_dict
        ss_model = DPRNNTasNet.from_pretrained(upload_state_dict(ss_ckpt))
  elif ss_name == "SuDoRMRF":
    raise NotImplementedError
  return ss_model

def main(conf):

    conf_id = conf["conf_id"]
    if conf["debug"]:
        exp_dir = os.path.join(conf["exp_dir"], "DEBUG")
    else:
        exp_dir = os.path.join(conf["exp_dir"], f"conf_id_{conf_id}", conf["job_id"])

    train_set = AMI_segments(
        split='train',
        csv_name='train_from_wavs.csv',
        n_src=conf["model"]["n_src"],
        use_sum_mixture=False # Controls whether the mixtures contains also the silent speakers or not
    )

    val_set = AMI_segments(
        split='valid',
        csv_name='valid_from_wavs.csv',
        n_src=conf["model"]["n_src"],
        use_sum_mixture=False # Controls whether the mixtures contains also the silent speakers or not
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    # Example of input for tensorboard
    batch_example = [val_set[i] for i in range(5)]

    model = load_ss_model(ss_name=conf["model"]["ss_model_name"], 
                          ss_ckpt=conf["model"]["ss_model_ckpt"])

    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["lr_scheduler"] == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer=optimizer, 
                                      factor=conf["training"]["lr_scheduler_args"]["factor"], 
                                      patience=conf["training"]["lr_scheduler_args"]["patience"])
    else:
       raise NotImplementedError

    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    if conf["training"]["loss"] == "si_sdr":
        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    else:
        raise ValueError("Loss {} not supported!".format(conf["training"]["loss"]))

    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
        batch_example=batch_example,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, 
        monitor=conf["training"]["model_ckpt_args"]["monitor"],
        mode=conf["training"]["model_ckpt_args"]["mode"],
        save_top_k=conf["training"]["model_ckpt_args"]["save_top_k"],
        verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor=conf["training"]["early_stop_args"]["monitor"], 
                                       mode=conf["training"]["early_stop_args"]["mode"], 
                                       patience=conf["training"]["early_stop_args"]["patience"], 
                                       verbose=True))

    trainer = pl.Trainer(
        max_epochs=conf["training"]["max_epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=conf["training"]["strategy"],
        devices=conf["training"]["devices"],
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=conf["training"]["gradient_clip_val"],
        #resume_from_checkpoint=conf["model"]["ss_model_ckpt"],
        logger=TensorBoardLogger(exp_dir, 
                                 log_graph=True,
                                 default_hp_metric=False),
        log_every_n_steps=1
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    #to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
  args = parser.parse_args()
  args = vars(args)

  conf = merge_dicts(common_parameters, conf[args["conf_id"]])
  conf = merge_dicts(conf, args)

  assert conf["model"]["ss_model_name"] == "ConvTasNet"
  if conf["model"]["ss_model_ckpt"] is None:
    sys.path.append("../")
    from final_models import ssep_models_dict
    conf["model"]["ss_model_ckpt"] = ssep_models_dict\
                                      [conf["model"]["sample_rate"]]\
                                      [conf["model"]["n_src"]]

  # What System expects
  conf["data"] = dict()
  conf["data"]["sample_rate"] = conf["model"]["sample_rate"]

  pprint(conf)
  main(conf)
