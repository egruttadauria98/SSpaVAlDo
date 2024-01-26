"""
Wrapper around pyannote-database to manage datasets for training and eval.
The staticmethods are a simpler interface to many pyannote-core functionalities.
Author: Elio Gruttadauria
"""

import os
import random

import soundfile as sf

from pyannote.core import Segment, SlidingWindowFeature, SlidingWindow
from diart.blocks.utils import Binarize

TESTED_PYANNOTE_DB_VERSIONS = ['5.0.0', '5.0.1']
DB_YML_FILES = {
  "AMI": {"yml": "..../AMI-diarization-setup/pyannote/database.yml",
          "sample_rate": 16000},
}

class PyannoteDataProtocol:
  """Wrapper to load any dataset setup as a Pyannote DB.

  Usage
  -----
  >>> from speaker_diarization.data.pyannote_data_protocol import PyannoteDataProtocol
  >>> pyannote_data = PyannoteDataProtocol(db_name="AMI")
  >>> pyannote_data.load_protocol(protocol_name="AMI.SpeakerDiarization.only_words")
  """
  def __init__(self, 
               db_name, # ex. 'AMI'
               old_loader=False):
    self.db_name = db_name
    self.old_loader = old_loader
    self._get_yml_path() #-> self.yml_path
    self._get_sample_rate() #-> self.sample_rate

    if self.old_loader:
      print("Using the standard loading strategy.\
           \nATTENTION: this loader does not work in training scripts.\
           \nIn general, files and DBs might not be found.")

  def _check_pyannote_db_version(self):
    # This check happens only when loading the data protocol
    import pkg_resources
    package_name = "pyannote.database"
    installed_version = pkg_resources.get_distribution(package_name).version
    if not installed_version in TESTED_PYANNOTE_DB_VERSIONS:
      raise ValueError(f"Pyannote database version {installed_version} is not tested."
                       "Please either: "
                       f"\t- Use one of the following versions: {TESTED_PYANNOTE_DB_VERSIONS}"
                       f"\t- Test {self.__class__.__name__} with your version and update the list of tested versions.")
    
  def _get_yml_path(self):
    try:
      self.yml_path = DB_YML_FILES[self.db_name]["yml"]
      if self.old_loader:
        os.environ["PYANNOTE_DATABASE_CONFIG"] = self.yml_path
    except KeyError:
      raise KeyError(f"Yaml file for database {self.db_name} not found in {DB_YML_FILES.keys()}"
                     "Please setup the yaml file and add the path to the DB_YML_FILES dict.")

  def _get_sample_rate(self):
    try:
      self.sample_rate = DB_YML_FILES[self.db_name]["sample_rate"]
    except KeyError:
      raise KeyError(f"Yaml file for database {self.db_name} not found in {DB_YML_FILES.keys()}"
                     "Please setup the yaml file and add the path to the DB_YML_FILES dict.")

  def load_protocol(self, 
                    protocol_name): # ex.'AMI.SpeakerDiarization.only_words'
    self._check_pyannote_db_version()
    
    from pyannote.database import FileFinder

    if self.old_loader:
      from pyannote.database import registry
      preprocessors = {'audio': FileFinder()}
      self.protocol = registry.get_protocol(protocol_name, preprocessors=preprocessors)
    else:
      from pyannote.database.registry import Registry
      registry = Registry()
      registry.load_database(self.yml_path)
      preprocessors = {'audio': FileFinder(registry=registry)}
      self.protocol = registry.get_protocol(protocol_name, preprocessors=preprocessors)
      #return preprocessors

  def get_split_iterator(self, 
                         split):
    """
    Return iterator over the given split of the given pyannote protocol.
    The returned iterator needs to be called.
    
    Example
    -------
    >>> test_set = pyannote_data.get_split_iterator("test")
    >>> for file in test_set():
    >>>   print(file['uri'])
    """
    assert split in ["train", "dev", "test"]
    if split == "train":
      return self.protocol.train
    elif split == "dev":
      return self.protocol.development
    elif split == "test":
      return self.protocol.test

  @staticmethod
  def crop(annotation, start_sec, end_sec):
    """Crop an annotation between start and end in seconds"""
    return annotation.crop(Segment(start=start_sec, end=end_sec))

  @staticmethod
  def generate_random_crop_file(file, segment_size, return_cropped_sig=False):
    """The input `file` is what is returned by the split iterator."""
    # Get path file
    path = file['audio']
    # Compute length in samples
    meeting_sig, meeting_sr = sf.read(path, dtype="float32")
    meeting_len = len(meeting_sig)
    seg_len_samples = int(segment_size * meeting_sr)
    # Sample random start and compute stop based on segment_size in samples
    start = random.randint(0, meeting_len - seg_len_samples)
    stop = start + seg_len_samples
    # Convert to seconds
    start_sec = start / meeting_sr
    stop_sec = stop / meeting_sr
    # Crop also annotations
    annotation_crop = PyannoteDataProtocol.crop(annotation=file['annotation'],
                                                start_sec=start_sec,
                                                end_sec=stop_sec)
    # TODO: this info can be extracted from the annotation crop anyway, no?
    info_crop = {
        'start_sec': start_sec,
        'start_sample': start,
        'stop_sec': stop_sec,
        'stop_sample': stop
    }
    if return_cropped_sig:
      return annotation_crop, info_crop, (meeting_sig[start: stop], meeting_sr)
    return annotation_crop, info_crop

  @staticmethod
  def discretize(annotation, start_sec, end_sec, resolution=0.005):
    # TODO: test unexpected behaviour without adding the support
    """Discretize and return data.
    By default it should return only the data where speech happens...
    """
    support = Segment(start=start_sec, end=end_sec)
    return annotation.discretize(support=support, resolution=resolution).data

  @staticmethod
  def get_speakers_and_duration_descending(annotation):
    """Retuns list of tuples with len 2.
    The first element is the speaker label, the second the duration
    >>> [('MIO082', 623.1500000000005),
         ('MIO050', 195.8099999999991),
         ('MIO016', 176.30999999999963),
         ('FIE081', 120.63000000000031)]
    """
    return annotation.chart()

  @staticmethod
  def get_speakers(annotation):
    """Returns list of labels of speakers.
    >>> ['FIE081', 'MIO016', 'MIO050', 'MIO082']
    """
    return annotation.labels()

  @staticmethod
  def get_num_speakers(annotation):
    return len(PyannoteDataProtocol.get_speakers(annotation))

  @staticmethod
  def get_speakers_sorted(annotation, descending=True, return_durations=True):
    speakers_and_durations = PyannoteDataProtocol.get_speakers_and_duration_descending(annotation)
    speakers, durations = zip(*speakers_and_durations)
    speakers = list(speakers)
    durations = list(durations)
    if not descending:
      speakers.reverse()
      durations.reverse()
    if return_durations:
      return speakers, durations
    else:
      return speakers

  @staticmethod
  def get_uem(annotation, overlap_only=False):
    """Uem is always timeline???"""
    # TODO: need to test this function, fundamental in evaluating individual segments!!!
    if overlap_only:
      return annotation.get_overlap() # should add here extend as well?
    # TODO: add meeting_reference.label_timeline('MIO082')
    # TODO: integrate with get_subset_speaker_annotation to get uem without overlapped speech?
    return annotation.get_timeline().extent()

  @staticmethod
  def get_subset_speaker_annotation(annotation, 
                                    speaker_label, 
                                    include_overlap=True,
                                    invert=False):
    # TODO: test it works without passing a subset
    #meeting_reference.subset(set(['MIO082']))
    """Returns annotation of a subset of speakers"""
    sub_annotation = annotation.subset(set([speaker_label]), invert)
    if include_overlap:
      return sub_annotation
    # TODO: test what happens when no overlap and invert true?
    uem_overlap = PyannoteDataProtocol.get_uem(overlap_only=True)
    return sub_annotation.extrude(annotation.get_overlap(), mode="intersection")

  @staticmethod
  def binarize_activities(est_activities, 
                          segment_size, 
                          start_sec, 
                          tau_active,
                          return_continuous=False):
    """Binarize estimatated activities.
    
    Parameters
    ----------
    est_activities : numpy array
      Shape (frames, speakers)
    """
    resolution = segment_size / est_activities.shape[0]
    out_continuous = SlidingWindowFeature(
        est_activities,
        SlidingWindow(start=start_sec,
                      duration=resolution, 
                      step=resolution)
    )
    binarize = Binarize(tau_active)
    out_binary = binarize(out_continuous)
    if return_continuous:
      return out_binary, out_continuous
    return out_binary

    



    
