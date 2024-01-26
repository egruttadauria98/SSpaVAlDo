"""
Utility functions for speech separation models.
"""

import torch

def split_estimated_sources(est_sources):
  """
  Split estimated sources into a list of tensors.

  This function assumes that the input tensor `est_sources` has the shape
  [batch_size, num_sources, samples], e.g., [32, 2, 80000]. It returns a list
  of tensors where each tensor has the shape [batch_size, 1, samples].

  Parameters
  ----------
  est_sources : torch.Tensor
      Input tensor with shape [batch_size, num_sources, samples].

  Returns
  -------
  list of torch.Tensor
      A list of tensors, each of shape [batch_size, 1, samples], representing
      each estimated source.
  """
  sources = [s.unsqueeze(1) for s in est_sources.transpose(0, 1)]
  return sources

def normalize_estimated_sources_batch(batch_input, est_sources):
  """
  Normalization of the audio sources, scaled by the ratio between
  the maximum absolute value of the input and the maximum absolute 
  of the estimated sources.
  
  ATTENTION: the maximum values are computed for the whole batch, not
  input by input.  

  This function assumes that the input tensor `est_sources` has the shape
  [batch_size, num_sources, samples], e.g., [32, 2, 80000]. It returns a list
  of tensors where each tensor has the shape [batch_size, 1, samples].

  Parameters
  ----------
  est_sources : torch.Tensor
      Input tensor with shape [batch_size, num_sources, samples].

  Returns
  -------
  list of torch.Tensor
      A list of tensors, each of shape [batch_size, 1, samples], representing
      each estimated source, after normalization.
  """
  sources = split_estimated_sources(est_sources)

  # Compute maximum absolute value in the batch,
  #  i.e. scalar
  wave_max = torch.max(torch.abs(batch_input))
  normalized_sources = []

  for source in sources:
      # Compute maximum absolute value for the estimated sources in the batch,
      #  i.e. scalar
      source_max = torch.max(torch.abs(source))
      # Normalize the source
      normalized_source = source * (wave_max / source_max)
      normalized_sources.append(normalized_source)

  return normalized_sources


def load_ss_model(ss_name, ss_ckpt):
  assert ss_name in ["ConvTasNet", "DPRNN", "SuDoRMRF"]
  if ss_name == "ConvTasNet":
    from asteroid.models import ConvTasNet
    try:
        ss_model = ConvTasNet.from_pretrained(ss_ckpt)
    except ValueError:
        from speaker_diarization.utils.torch_utils import upload_state_dict
        ss_model = ConvTasNet.from_pretrained(upload_state_dict(ss_ckpt))
  elif ss_name == "DPRNN":
    from asteroid.models import DPRNNTasNet
    try:
        ss_model = DPRNNTasNet.from_pretrained(ss_ckpt)
    except ValueError:
        from speaker_diarization.utils.torch_utils import upload_state_dict
        ss_model = DPRNNTasNet.from_pretrained(upload_state_dict(ss_ckpt))
  elif ss_name == "SuDoRMRF":
    raise NotImplementedError
  return ss_model
