"""
Online diarization tuning script.
"""

import os

from speaker_diarization.inference.online.diart_wrapper import DiartWrapper, ModelLoader
from speaker_diarization.inference.online.config import DiartConfig
from speaker_diarization.utils.hf_utils import hf_token
from speaker_diarization.utils.print_utils import print_args_decorator

from diart.optim import Optimizer
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.models import EmbeddingModel

@print_args_decorator
def run_tune(model,
             db_name,
             params,
             output_dir,
             split="dev",
             embedding_model_name="pyannote/embedding",
             sample_rate=16000,
             duration=5,
             step=0.5,
             latency=5,
             max_iters=100):
  """
  Run online diarization in batch mode, different from sequential real-time.

  Parameters
  ----------
  model : Pytorch model
    Model to perform inference in the local window. 
    Used with the diart wrapper.
  db_name : str
    Name of the dataset. Must match one of the keys in config.py
    ex. AMI
  params : str or dict
    If str, must match one the default params in config.py
    If dict, must specify all the parameters, as done in config.py
    In the tuning, the params act as a seed for the Optuna search.
  output_dir : str
    Path to the directory where the output rttm files will be saved.
  embedding_model_name : str
    Name of the embedding model on Hugging Face.
  sample_rate: int
    Sample rate of the data in audio_dir
  duration: float
    Duration of the local window in seconds.
    This will represent the atomic unit for the stitching mechanism.
  step: float
    Step size in seconds between two consecutive local windows.
    Also known as the hop size.
  latency: float
    Latency in seconds of the system.
    The minimum value is the step size, and the maximum value is the duration.
    Any value from the minimum, at increments of the step size, up to the maximum is valid.
    Please refer to https://arxiv.org/abs/2109.06483 for more details.
  max_iters: int
    Iterartions of the tuning process.
    # TODO: check if one iter = all dataset
  """

  # STEP 0: Init config as seed for tuning
  diart_config = DiartConfig(db=db_name,
                             params=params,
                             split=split) # Always tune on dev split
  
  # STEP 1: WRAP THE MODEL
  segmentation_model = DiartWrapper(ModelLoader(model),
                                    sample_rate=sample_rate,
                                    duration=duration)

  # STEP 2: GET EMBEDDING MODEL
  embedding_model = EmbeddingModel.from_pyannote(embedding_model_name,
                                                 use_hf_token=hf_token())

  # STEP 3: DEFINE DIART OBJECTS
  config = SpeakerDiarizationConfig(
      segmentation=segmentation_model,
      embedding=embedding_model,
      step=step,
      latency=latency,
      tau_active=diart_config.params["tau_active"],
      rho_update=diart_config.params["rho_update"],
      delta_new=diart_config.params["delta_new"],
  )

  # STEP 4: RUN OPTIMIZATION
  optimizer_dir = os.path.join(output_dir, "tuning_hyperparameters_diart")
  os.makedirs(optimizer_dir, exist_ok=True)
  optimizer = Optimizer(SpeakerDiarization,
                        diart_config.audio_dir,
                        diart_config.rttm_dir,
                        optimizer_dir,
                        base_config=config)
  optimizer(num_iter=max_iters)

  best_params = dict()
  for name, value in optimizer.study.best_params.items():
    print(f"\n{name}: {value}")
    if name in ["tau_active", "rho_update" ,"delta_new"]:
      best_params[name] = value
  
  return best_params

