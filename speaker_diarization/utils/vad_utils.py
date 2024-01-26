"""
Utility functions for VAD models.
"""

def load_vad_model(vad_name, vad_ckpt):
  if "pyannote" in vad_name:
    from pyannote.audio import Model
    from speaker_diarization.utils.hf_utils import hf_token
    vad_model = Model.from_pretrained(vad_ckpt, 
                                      use_auth_token=hf_token())
  elif "name" in vad_name:
    raise NotImplementedError
  return vad_model