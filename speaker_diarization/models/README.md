# Models



### To do:
1. If the VAD is from Pyannote, then it is actually a `Segmentation` model that performs speaker diarization. As such the number of output channels need to be reduced from 3 or 4 to 1 (ex. [batch_size, time, 3] -> [batch_size, time, 1]). Currently, the code follows the same approach from Pyannote to use a `max_per_frame` strategy, meaning that the VAD output is the max speech activation per frame. Another strategy that could be studies in the future is `max_on_average`, which takes only one of the speaker channels. This change in strategy might be beneficial for E2E training.
  - `max_per_frame` : `[torch.amax(output, dim=2, keepdim=True) for output in vad_outputs]`
  - `max_on_average` : `[output[torch.arange(output.size(0)).unsqueeze(1).unsqueeze(2), torch.arange(output.size(1)).unsqueeze(0).unsqueeze(2), torch.argmax(torch.mean(output, dim=1), dim=1).unsqueeze(1).unsqueeze(2)] for output in vad_outputs]`
where `vad_outputs` is a list of length n_outputs_ss, e.g., if n_outputs_ss = 2, it contains 2 tensors. Each tensor shape: [batch_size, 293, 3], e.g., [32, 293, 3]