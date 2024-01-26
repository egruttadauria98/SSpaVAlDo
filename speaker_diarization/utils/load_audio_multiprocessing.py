"""
Utility functions for loading audio with multiple processes.
"""

import torch
import torchaudio
from multiprocessing import Pool

def load_segment_torchaudio(args, return_sr=False):
    filename, start, stop = args
    waveform, sample_rate = torchaudio.load(filename, num_frames=stop-start, frame_offset=start)
    if return_sr:
        return waveform, sample_rate
    return waveform

# TODO: benchmark if actually faster on long recordings
def load_audio_multiprocessing_torchaudio(audio_path, num_processes):
    # Get the length of the audio file without loading it
    audio_info = torchaudio.info(audio_path)
    length = audio_info.num_frames

    # Calculate segment lengths
    segment_length = length // num_processes
    segments = []

    # Create arguments for each process
    for i in range(num_processes):
        start = i * segment_length
        # Make the last segment extend to the end of the file
        stop = (i+1) * segment_length if i != num_processes - 1 else length
        segments.append((audio_path, start, stop))

    # Load audio segments
    with Pool(num_processes) as p:
        audio_segments = p.map(load_segment_torchaudio, segments)

    from load_audio import load_audio_torchaudio
    _, sample_rate = load_audio_torchaudio(audio_path, 0, 1)

    return torch.cat(audio_segments, dim=-1), sample_rate