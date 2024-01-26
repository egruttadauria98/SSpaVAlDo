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
      "vad_model_name": "pyannote_3out",
      "vad_model_ckpt": None, # If None, filled automatically in train.py
      "vad_max_method": "max_per_frame", # Defined only if VAD is Pyannote
      "ss_vad_model_ckpt": None, # If None, test only pretrained SSep and VAD models
      "embedding_model": "pyannote/embedding"
      },
    "data": {
      "db_name": "AMI",
      },
    "tune": {
      "skip": False,
      "params": "diart_all_latency", #IMPORTANT: if str and not dict, this is specific to the db_name
      "duration": 5,
      "step": 0.5,
      "latency": 5,
      "max_iters": 100,
      "split": "dev",
    },
    "benchmark": {
      "skip": False,
      "params": "best_from_tune",
      "duration": None, # If None, set the same as in tune
      "step": None,
      "latency": None,
      "split": "test",
      "batch_size": 32,
    },
    "exp_dir": exp_dir,
    "job_id": job_id,
    "cluster": cluster,
}

conf = {
# PRETRAINED SSep and VAD models
  # N_SRC = 2
    # MAX LATENCY
  "001": {
    "model": {
      "n_src": 2,
      "embedding_model": "pyannote/embedding"
      },
    "tune": {
      "params": "diart_all_latency", #IMPORTANT: if str and not dict, this is specific to the db_name
      "duration": 5,
      "step": 0.5,
      "latency": 5,
    }
    },
    # MIN LATENCY
  "002": {
    "model": {
      "n_src": 2,
      "embedding_model": "pyannote/embedding"
      },
    "tune": {
      "params": "diart_all_latency", #IMPORTANT: if str and not dict, this is specific to the db_name
      "duration": 5,
      "step": 0.5,
      "latency": 0.5,
    }
    },
  # N_SRC = 3
    # MAX LATENCY
  "003": {
    "model": {
      "n_src": 3,
      "embedding_model": "pyannote/embedding"
      },
    "tune": {
      "params": "diart_all_latency", #IMPORTANT: if str and not dict, this is specific to the db_name
      "duration": 5,
      "step": 0.5,
      "latency": 5,
    }
    },
    # MIN LATENCY
  "004": {
    "model": {
      "n_src": 3,
      "embedding_model": "pyannote/embedding"
      },
    "tune": {
      "params": "diart_all_latency", #IMPORTANT: if str and not dict, this is specific to the db_name
      "duration": 5,
      "step": 0.5,
      "latency": 0.5,
    }
    },
# TRAINED SSepVAD
  # N_SRC = 2
    # MAX LATENCY
  "005": {
    "model": {
      "n_src": 2,
      "ss_vad_model_ckpt": "",
      "embedding_model": "pyannote/embedding"
      },
    "tune": {
      "params": "diart_all_latency", #IMPORTANT: if str and not dict, this is specific to the db_name
      "duration": 5,
      "step": 0.5,
      "latency": 5,
    }
    },
    # MIN LATENCY
  "006": { 
    "model": {
      "n_src": 2,
      "ss_vad_model_ckpt": "",
      "embedding_model": "pyannote/embedding"
      },
    "tune": {
      "params": "diart_all_latency", #IMPORTANT: if str and not dict, this is specific to the db_name
      "duration": 5,
      "step": 0.5,
      "latency": 0.5,
    }
    },
  # N_SRC = 3
    # MAX LATENCY
  "007": {
    "model": {
      "n_src": 3,
      "ss_vad_model_ckpt": "",
      "embedding_model": "pyannote/embedding"
      },
    "tune": {
      "params": "diart_all_latency", #IMPORTANT: if str and not dict, this is specific to the db_name
      "duration": 5,
      "step": 0.5,
      "latency": 5,
    }
    },
    # MIN LATENCY
  "008": { 
    "model": {
      "n_src": 3,
      "ss_vad_model_ckpt": "",
      "embedding_model": "pyannote/embedding"
      },
    "tune": {
      "params": "diart_all_latency", #IMPORTANT: if str and not dict, this is specific to the db_name
      "duration": 5,
      "step": 0.5,
      "latency": 0.5,
    }
    },
}
