"""
Utility functions for torch.
"""

import torch

def upload_state_dict(ckpt):
  # Convert .ckpt into .pth
  try:
      state_dict = torch.load(ckpt)
  except RuntimeError:
      # This case for no GPU
      state_dict = torch.load(ckpt, map_location=torch.device('cpu'))
  return state_dict['state_dict']