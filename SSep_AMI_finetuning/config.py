"""
This is a configuration file used to forward desired parameters to
the different modules used during training.
"""

import os
import random
import socket
import string

# CLUSTER
cluster = socket.gethostname()
slurm = "SLURM_JOB_ID" in os.environ

# STORAGE PATH
exp_dir = ""

# Assign job_id to the run
if slurm:
  job_id = os.environ["SLURM_JOB_ID"]
else:
  job_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))

common_parameters = {
    "model": {
      "sample_rate": 16000,
      "n_src": 2,
      "ss_model_name": "ConvTasNet",
      "ss_model_ckpt": None, # If None, filled automatically in train.py
      },
    "optim": {
        "lr": 1e-4,
        "weight_decay": 0.0},
    "training": {
        "loss": "si_sdr",
        "max_epochs": 500,
        "batch_size": 8,
        "num_workers": int(os.environ["SLURM_CPUS_ON_NODE"]) if slurm else 10,
        "model_ckpt_args": {"monitor": "val_loss",
                            "mode": "min",
                            "save_top_k": 5},
        "early_stop": True,
        "early_stop_args": {"monitor": "val_loss", 
                            "mode": "min",
                            "patience": 30},
        "gradient_clip_val": 5.0,
        "lr_scheduler": "ReduceLROnPlateau",
        "lr_scheduler_args": {"factor": 0.5, 
                              "patience": 5},
        #"lr_scheduler": "OneCycleLR",
        #"lr_scheduler_args": {"max_lr": 1e-3, 
        #                      "total_steps": None,
        #                      "div_factor": 10, 
        #                      "pct_start": 0.15},
        "wd_scheduler": False,
        #"wd_scheduler_args": {"wd_min": 1e-12, 
        #                      "wd_max": 0.05,
        #                      "total_steps": None},
        "batch_scheduler": False,
        #"batch_sizes": [16, 24, 32, 64, 128], # for the batch scheduler
        #"epochs": [1, 2, 5, 10, 20], # for the batch scheduler
        "strategy": "ddp",
        "devices": "auto"},
    "exp_dir": exp_dir,
    "job_id": job_id,
    "cluster": cluster
}

conf = {
  "001": {
    "model": {
      "sample_rate": 16000,
      "n_src": 2,
      },
    },
  "002": {
    "model": {
      "sample_rate": 16000,
      "n_src": 3,
      },
    "training": {
      "batch_size": 2,
      }, 
  }

}
