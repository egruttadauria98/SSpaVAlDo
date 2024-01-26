import random

STORAGE_PARAMS = {
  "AMI": {
    "diart_all_latency":{
      "tau_active": 0.507,
      "rho_update": 0.006,
      "delta_new": 1.057,
    }
  }
}

STORAGE_DB = {
  "AMI": {
    "dev":{
      "audio_dir": "..../AMI_audio_dev",
      "rttm_dir": "..../AMI-diarization-setup/word_and_vocalsounds/rttms/dev"
    },
    "test": {
      "audio_dir": "..../AMI_audio_test",
      "rttm_dir": "..../AMI-diarization-setup/word_and_vocalsounds/rttms/test"
    }
  }
}

def get_diart_empty_config_dict():
  d = {
    "tau_active": None,
    "rho_update": None,
    "delta_new": None,
  }
  return d

class DiartConfig:

  # Params always stored as dict, not individual attributes
  def __init__(self, 
               db, #ex. AMI 
               params, # either str or dict
               split, #ex. dev
               ): 
    
    self.db = db
    self.split = split
    self.audio_dir, self.rttm_dir = self._get_saved_paths()

    if isinstance(params, str):
      params_str = params
      if params_str == "random":
        self.params = self._set_random_config()
      else:
        self.params = self._get_saved_params(params_str)
    elif isinstance(params, dict):
      self.params = params
    else:
      raise ValueError("params must be either str or dict")

  # NOT USED
  #def _check_empty_dict(self, config_dict):
  #  # Return true if at least one parameter is None
  #  for _, v in config_dict.items():
  #    if v is None:
  #      return True

  def _set_random_config(self):
    # Sampling range based on
    # https://github.com/juanmc2005/diart/blob/main/src/diart/optim.py#L33

    config_dict = get_diart_empty_config_dict()
    config_dict["tau_active"] = random.uniform(0, 1)
    config_dict["rho_update"] = random.uniform(0, 1)
    config_dict["delta_new"] = random.uniform(0, 2) # Why 2 is max?
    return config_dict

  def _get_saved_params(self, params_str):
    try:
      db_params = STORAGE_PARAMS[self.db]
      try:
        params = db_params[params_str]
      except KeyError:
        raise KeyError(f"Params {params} not in {self.db} STORAGE_PARAMS")
    except KeyError:
      raise KeyError(f"Database {self.db} not in STORAGE_PARAMS")
    return params
  
  def _get_saved_paths(self):
    try:
      db_paths = STORAGE_DB[self.db]
      try:
        paths = db_paths[self.split]
      except KeyError:
        raise KeyError(f"Split {self.split} not in {self.db} STORAGE_DB")
    except KeyError:
      raise KeyError(f"Database {self.db} not in STORAGE_DB")
    return paths['audio_dir'], paths['rttm_dir']
