"""
Script to tune the hyperparams for Diart online inference.
"""

import os
import sys
import argparse
from pprint import pprint
from config_online import common_parameters, conf

import torch

from speaker_diarization.utils.ssep_utils import load_ss_model
from speaker_diarization.utils.vad_utils import load_vad_model
from speaker_diarization.models.pyannote_ss_vad_model import SS_VAD_model
from speaker_diarization.utils.print_utils import print_args_decorator
from speaker_diarization.utils.train_utils import merge_dicts

from speaker_diarization.inference.online.tune import run_tune
from speaker_diarization.inference.online.benchmark import run_benchmark

parser = argparse.ArgumentParser()
parser.add_argument("--conf_id", required=True,
                    help="Conf tag, used to get the right config")
parser.add_argument("--debug", type=bool, default=False,
                    help="If true save to specific directory")

@print_args_decorator
def main(conf):

  conf_id = conf["conf_id"]
  if conf["debug"]:
    exp_dir = os.path.join(conf["exp_dir"], "DEBUG")
  else:
    exp_dir = os.path.join(conf["exp_dir"], f"conf_id_{conf_id}", conf["job_id"])
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("\nDevice: ", device)

  # Load SS model
  ss_model = load_ss_model(ss_name=conf["model"]["ss_model_name"],
                           ss_ckpt=conf["model"]["ss_model_ckpt"])

  # Load VAD model
  vad_model = load_vad_model(vad_name=conf["model"]["vad_model_name"],
                             vad_ckpt=conf["model"]["vad_model_ckpt"])

  if conf["model"]["ss_vad_model_ckpt"] is None:
    # Assemble SS-VAD model
    model = SS_VAD_model(ss_model,
                        vad_model,
                        vad_max_method=conf["model"]["vad_max_method"])
  else:
    # Load best checkpoint of the trained model
    model = SS_VAD_model.load_from_checkpoint(conf["model"]["ss_vad_model_ckpt"],
                                              ss_model=ss_model,
                                              vad_model=vad_model,
                                              map_location=torch.device(device))

  model.eval()

  if not conf["tune"]["skip"]:
    best_params = run_tune(
          model=model,
          db_name=conf["data"]["db_name"],
          params=conf["tune"]["params"],
          output_dir=exp_dir,
          embedding_model_name=conf["model"]["embedding_model"],
          sample_rate=conf["model"]["sample_rate"],
          duration=conf["tune"]["duration"],
          step=conf["tune"]["step"],
          latency=conf["tune"]["latency"],
          max_iters=conf["tune"]["max_iters"]
      )

  if not conf["benchmark"]["skip"]:

    if conf["benchmark"]["params"] == "best_from_tune":
      conf["benchmark"]["params"] = best_params
    else:
      raise NotImplementedError("Only best_from_tune is implemented")

    run_benchmark(
          model=model,
          db_name=conf["data"]["db_name"],
          params=conf["benchmark"]["params"],
          output_dir=exp_dir,
          embedding_model_name=conf["model"]["embedding_model"],
          sample_rate=conf["model"]["sample_rate"],
          duration=conf["benchmark"]["duration"],
          step=conf["benchmark"]["step"],
          latency=conf["benchmark"]["latency"],
          bs=conf["benchmark"]["batch_size"]
      )

if __name__ == "__main__":
  args = parser.parse_args()
  args = vars(args)

  conf = merge_dicts(common_parameters, conf[args["conf_id"]])
  conf = merge_dicts(conf, args)

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

  for param in ["duration", "step", "latency"]:
    if conf["benchmark"][param] is None:
      conf["benchmark"][param] = conf["tune"][param]

  pprint(conf)
  main(conf)
