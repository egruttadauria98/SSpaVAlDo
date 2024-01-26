import torch

from pyannote.audio import Model
from pyannote.audio.core.task import Specifications

from speaker_diarization.utils.ssep_utils import normalize_estimated_sources_batch

# Follow PyanNet which hinerits from Model
# https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/models/segmentation/PyanNet.py

class SS_VAD_model(Model):
    """
    This class inherits from Pyannote's Model class, which itself inherits from PL.
    As such, this model is easily integrable with other Pyannote objects, such as Tasks.
    The model given as input do NOT need to be Pyannote models.
    """
    def __init__(self, ss_model, vad_model, ss_freeze=True, vad_max_method="max_per_frame"):
      super(SS_VAD_model, self).__init__()
      self.ss_model = ss_model
      self.vad_model = vad_model
      self.vad_max_method = vad_max_method

      # Check if the VAD model is a Pyannote model
      self.vad_is_pyannote = isinstance(self.vad_model, Model)

      # Allow to run print only on the first batch for the forward method
      self.print_first_batch = True
      
      if ss_freeze:
        for param in self.ss_model.parameters():
          param.requires_grad = False

    def setup_specifications(self, duration=5):
      self.specifications = Specifications(
            problem=self.vad_model.specifications.problem,
            resolution=self.vad_model.specifications.resolution,
            duration=float(duration),
            warm_up=self.vad_model.specifications.warm_up,
            classes=self.vad_model.specifications.classes,
        )
      
    def pyannote_vad_output(self, vad_output, strategy="max_per_frame"):
      if strategy == "max_per_frame":
        vad_output = [torch.amax(output, dim=2, keepdim=True) for output in vad_output]
      elif strategy == "max_on_average":
        raise NotImplementedError
      return vad_output

    def forward(self, 
                waveforms, # Input: [batch_size, 1, samples], e.g., [32, 1, 80000]
                return_sources=False): 

      est_sources = self.ss_model(waveforms) # Output: [batch_size, n_outputs_ss, samples], e.g., [32, 2, 80000]

      # normalized_sources is a list of length n_outputs_ss, 
      # e.g., if n_outputs_ss = 2, it contains 2 tensors.
      # Each tensor shape: [batch_size, 1, samples], e.g., [32, 1, 80000]
      # sources content: List of tensors representing estimated sources.
      normalized_sources = normalize_estimated_sources_batch(batch_input=waveforms, 
                                                             est_sources=est_sources)

      # List of tensors of shape [batch_size, resolution, n_classes]
      # For Pyannote VAD, each tensor shape: [batch_size, 293, 3], e.g., [32, 293, 3]
      vad_output = [self.vad_model(source) for source in normalized_sources]
      
      if self.vad_is_pyannote:
        # Each tensor has the last dimension pooled to 1
        vad_output = self.pyannote_vad_output(vad_output)

      # Stack the VAD estimated on each separated source
      # Shape: [batch_size, resolution, n_outputs_ss], e.g., [32, 293, 2]
      out = torch.stack(vad_output, dim=2).squeeze(-1) 

      if self.print_first_batch:
        # Do not print for safety check, only for the first real batch
        if not waveforms.shape[0] == 1:  
          print("\nINFO ON FORWARD METHOD:")          
          print("\twaveforms.shape", waveforms.shape)
          print("\test_sources.shape", est_sources.shape)
          print("\tlen(normalized_sources)", len(normalized_sources))
          print("\tnormalized_sources[0].shape", normalized_sources[0].shape)
          print("\tlen(vad_output)", len(vad_output))
          print("\tvad_output[0].shape", vad_output[0].shape)
          print("\tlen(vad_output) pt2", len(vad_output))
          print("\tvad_output[0].shape pt2", vad_output[0].shape)
          print("\tout.shape", out.shape)
          self.print_first_batch = False
      
      if return_sources:
        # Used for TensorBoard logging with enhanced task
        return out, normalized_sources
      return out

    