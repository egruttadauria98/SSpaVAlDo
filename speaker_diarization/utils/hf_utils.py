"""
Utility functions for HF.
"""

import os

def hf_token():
  hf_token = open(f"{os.environ['AUDIBLE_STORAGE_ROOT']}/speech/studies/speaker-segmentation/hf_token.txt", 'r').readlines()[0]
  return hf_token