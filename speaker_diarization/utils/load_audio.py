"""
Utility functions for loading audio.
"""

import torchaudio

def load_audio_torchaudio(filename, start=0, stop=None):
    if stop is None:
      num_frames = -1
    else:
      num_frames = stop-start
    waveform, sample_rate = torchaudio.load(filename, 
                                            num_frames=num_frames, 
                                            frame_offset=start)
    return waveform, sample_rate
