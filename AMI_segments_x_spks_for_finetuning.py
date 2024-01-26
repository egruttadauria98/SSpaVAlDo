import os
os.environ["PYANNOTE_DATABASE_CONFIG"] = '/path/to/AMI-diarization-setup/pyannote/database.yml'
from pyannote.database import FileFinder
preprocessors = {'audio': FileFinder()}

from pyannote.database import registry
only_words = registry.get_protocol('AMI.SpeakerDiarization.only_words', preprocessors=preprocessors)

import numpy as np
import soundfile as sf
import random
import pandas as pd

from pyannote.core import Segment

tot_iter_per_meeting = 75 # 300 for 2 spks, 75 for 3 spks
segment_size_sec = 35 # 5 for 2 spks, 35 for 3 spks
minimum_perc_in_segment = 0.05 # 0.15 for 2 spks, 0.05 for 3 spks

SPKS_SEGMENTS = 3

base_path = "/path/to/AMI-diarization-setup/AMI_individual_headset/amicorpus"
save_path = f"/path/to/AMI-diarization-setup/AMI_individual_headset/random_{SPKS_SEGMENTS}spks/wavs"

os.makedirs(save_path, exist_ok=True)

def order_spks_by_activity(reference):
  spk_list = [] 
  time_list = []
  for el in reference.chart():
    spk_list.append(el[0])
    time_list.append(el[1])
  return spk_list, time_list

def compute_sdr(target, estimate):

  # Ensure target and estimate have the same shape
  if target.shape != estimate.shape:
    raise ValueError("Target and Estimate must have the same shape.")

  # Compute the squared norms
  target_norm2 = np.sum(target ** 2)
  error_norm2 = np.sum((target - estimate) ** 2)

  # Compute the SDR
  sdr = 10 * np.log10(target_norm2 / error_norm2)
  
  return sdr

metadata_dict = dict()
metadata_dict['seg_name'] = []
metadata_dict['path_mixture'] = []

for i in range(SPKS_SEGMENTS):
  metadata_dict[f'path_source_{i+1}'] = []
  metadata_dict[f'original_source_{i+1}'] = []

metadata_dict['start'] = []
metadata_dict['stop'] = []

for file in only_words.train():

  try:

    uri = file['uri']
    print(uri)

    # Get the total length for the random segments
    meeting_all, sr = sf.read(file['audio'], dtype="float32")
    meeting_len_sec = len(meeting_all)/sr
    print(f"\tMeeting len: {meeting_len_sec} seconds")

    path_sources = os.path.join(base_path, f"{uri}/audio")
    print(f"\tTotal individual meetings: {len(os.listdir(path_sources))}\n")
    
    for i in range(tot_iter_per_meeting):

      spks = 0
      skip = False

      while spks != SPKS_SEGMENTS:
        # Randomly select a segment until it has 2 speakers
        start = random.randint(1, int(meeting_len_sec - segment_size_sec)-2)
        stop = start + segment_size_sec
        annotation_crop = file['annotation'].crop(Segment(start=start, end=stop))
        spks = len(annotation_crop.labels())

        if spks == SPKS_SEGMENTS:
          # Check least active speaker is more than 15 percent
          _, time_list = order_spks_by_activity(annotation_crop)
          if not time_list[-1] >= segment_size_sec * minimum_perc_in_segment:
            spks = 0

      seg_name = f"{uri}_{start}_{stop}"
      print(f"\tSegment {i+1} of {tot_iter_per_meeting}: {seg_name}")

      sdr_dict = dict()

      mix, _ = sf.read(file['audio'], dtype="float32", start=start*sr, stop=stop*sr)

      for s_name in os.listdir(path_sources):
        s_path = os.path.join(path_sources, s_name)
        s, _ = sf.read(s_path, dtype="float32", start=start*sr, stop=stop*sr)
        sdr_dict[s_name] = compute_sdr(s, mix)
      
      top_spks_keys = sorted(sdr_dict, key=sdr_dict.get, reverse=True)[:SPKS_SEGMENTS]

      if not skip:

        sf.write(os.path.join(save_path, f"{seg_name}_mixture.wav"), mix, sr)

        metadata_dict['seg_name'].append(seg_name)
        metadata_dict['path_mixture'].append(os.path.join(save_path, f"{seg_name}_mixture.wav"))
        metadata_dict['start'].append(start)
        metadata_dict['stop'].append(stop)

        # Read the segments from the sources
        for i, s_name in enumerate(top_spks_keys):
          s_path = os.path.join(path_sources, s_name)
          s, sr_s = sf.read(s_path, dtype="float32", start=start*sr, stop=stop*sr)
          sf.write(os.path.join(save_path, f"{seg_name}_{i}.wav"), s, sr)

          # Save the metadata
          metadata_dict[f'path_source_{i+1}'].append(os.path.join(save_path, f"{seg_name}_{i}.wav"))
          metadata_dict[f'original_source_{i+1}'].append(s_name)
  except Exception as e: 
    print(f"\n\nSKIPPING MEETING {file['uri']}\n\n")
    print(e)
    print()

df = pd.DataFrame(metadata_dict)
df.to_csv(os.path.join("/".join(save_path.split("/")[:-1]), "metadata.csv"), index=False)

# Extract unique seg_names and shuffle them
unique_segs = df['seg_name'].unique()
np.random.shuffle(unique_segs)

# Determine the number of unique seg_name values needed for an 80-10-10 split
train_size = int(0.8 * len(unique_segs))
test_size = int(0.1 * len(unique_segs))

train_segs = unique_segs[:train_size]
test_segs = unique_segs[train_size:train_size+test_size]
dev_segs = unique_segs[train_size+test_size:]

# Split the DataFrame based on seg_name values
train_df = df[df['seg_name'].isin(train_segs)]
test_df = df[df['seg_name'].isin(test_segs)]
dev_df = df[df['seg_name'].isin(dev_segs)]

train_df.to_csv(os.path.join("/".join(save_path.split("/")[:-1]), "metadata_train.csv"), index=False)
test_df.to_csv(os.path.join("/".join(save_path.split("/")[:-1]), "metadata_test.csv"), index=False)
dev_df.to_csv(os.path.join("/".join(save_path.split("/")[:-1]), "metadata_valid.csv"), index=False)
