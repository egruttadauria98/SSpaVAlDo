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
      "ss_freeze": True,
      "ss_model_name": "ConvTasNet",
      "ss_model_ckpt": None, # If None, filled automatically in train.py
      "vad_model_name": "pyannote_3out",
      "vad_model_ckpt": None, # If None, filled automatically in train.py
      "vad_max_method": "max_per_frame" # Defined only if VAD is Pyannote
      },
    "data": {
      "protocol_name": "AMI.SpeakerDiarization.only_words", #"AMI.SpeakerDiarization.mini",
      "segment": 5,
      "max_speakers_per_chunk": "n_src" # If "n_src", then train only on segments with at most n_src speakers
      },
    "loss": {
        "vad_loss": "bce" # Either "bce" or "mse" or None
      },
    "optim": {
        "lr": 1e-4,
        "weight_decay": 0.0},
    "training": {
        "max_epochs": 500,
        "batch_size": 8,
        "num_workers": int(os.environ["SLURM_CPUS_ON_NODE"]) if slurm else 10,
        "early_stop": True,
        "early_stop_min_delta": 0.0,
        "early_stop_patience": 10,
        "gradient_clip_val": 0.5,
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
        "devices": int(os.environ["SLURM_GPUS_ON_NODE"]) if "SLURM_GPUS_ON_NODE" in os.environ else 1,
        "num_nodes": int(os.environ["SLURM_NNODES"]) if slurm else 1},
    "exp_dir": exp_dir,
    "job_id": job_id,
    "cluster": cluster,
    "eval": {
        "metric_type": "DER",
        "split": "test",
        "segment_size": 5,
        "tau_list": [0.45, 0.5, 0.55], # # Check on Tensorboard if optimal threshold is outside of the range (very unlikely)
        "n_bootstrap_samples": 10,
        "n_iterations_per_sample": 25,
        "max_spks": 5, # Returns detailed DER on segments with only x speakers from 0 to max_spks
        "oracle_spks_num": False,
        "score_overlap_only": False,
    }
}

conf = {
  "001": { # FINETUNE ONLY VAD, with 2 outputs
    "data": {
      "max_speakers_per_chunk": "n_src" # If "n_src", then train only on segments with at most n_src speakers
      },
    "loss": {
        "vad_loss": "bce" # Either "bce" or "mse" or None
      },
    },

  "002": { # FINETUNE ONLY VAD, with 3 outputs
    "model": {
      "n_src": 3
     },  
    "data": {
      "max_speakers_per_chunk": "n_src" # If "n_src", then train only on segments with at most n_src speakers
      },
    "loss": {
        "vad_loss": "bce" # Either "bce" or "mse" or None
      },
    },

  "003": { # FINETUNE E2E, with 2 outputs
    "model": {
      "ss_freeze": False
     },
    "data": {
      "max_speakers_per_chunk": "n_src" # If "n_src", then train only on segments with at most n_src speakers
      },
    "loss": {
        "vad_loss": "bce" # Either "bce" or "mse" or None
      },
    },

  "004": { # FINETUNE E2E, with 3 outputs
    "model": {
      "ss_freeze": False,
      "n_src": 3
     },
    "data": {
      "max_speakers_per_chunk": "n_src" # If "n_src", then train only on segments with at most n_src speakers
      },
    "loss": {
        "vad_loss": "bce" # Either "bce" or "mse" or None
      },
    },

}
