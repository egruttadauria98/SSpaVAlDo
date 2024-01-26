import torch

from pyannote.audio import Model
from pyannote.audio.core.task import Specifications

# Follow PyanNet which hinerits from Model
# https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/models/segmentation/PyanNet.py

class PyannoteWrapper(Model):
    """
    This class inherits from Pyannote's Model class, which itself inherits from PL.
    As such, this model is easily integrable with other Pyannote objects, such as Tasks.
    The model given as input do NOT need to be Pyannote models.
    """
    def __init__(self, model, *args, **kwargs) -> None:
      super().__init__(*args, **kwargs)
      self.model = model

      # Allow to run print only on the first batch for the forward method
      self.print_first_batch = True
      
    #def setup_specifications(self, duration=5):
    #  self.specifications = Specifications(
    #        problem=self.vad_model.specifications.problem,
    #        resolution=self.vad_model.specifications.resolution, #How this affected by the same rate?
    #        duration=float(duration),
    #        warm_up=self.vad_model.specifications.warm_up,
    #        classes=self.vad_model.specifications.classes,
    #    )

    def forward(self, 
                waveforms):  # Input: [batch_size, 1, samples], e.g., [32, 1, 80000]

      print(waveforms.shape)
      out = self.model(waveforms)

      if self.print_first_batch:
        # Do not print for safety check, only for the first real batch
        if not waveforms.shape[0] == 1:  
          print("\nINFO ON FORWARD METHOD:")          
          print("\twaveforms.shape", waveforms.shape)
          print("\tout.shape", out.shape)
          self.print_first_batch = False
      
      return out
