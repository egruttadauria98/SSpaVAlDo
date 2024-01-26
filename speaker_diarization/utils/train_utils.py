"""
Utility functions for training.
"""

def merge_dicts(dict1, dict2):
  """ Recursively merge two dictionaries, with values from dict2 
  taking precedence.

  Parameters
  ----------
  dict1 : dict
    First dictionary.
  dict2 : dict
    Second dictionary.

  Returns
  -------
  dict
    The merged dictionary.
  """
  merged = dict1.copy()
  for key, value in dict2.items():
    if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
      merged[key] = merge_dicts(merged[key], value)
    else:
      merged[key] = value
  return merged