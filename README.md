# Introduction

This repository constains the code to reproduce the results of the paper ["ONLINE SPEAKER DIARIZATION OF MEETINGS GUIDED BY SPEECH SEPARATION"](https://hal.science/hal-04419041), accepted to ICASSP 2024.\
The code is divided into three main experiments:
1. `SSep_AMI_finetuning`: finetuning a pretrained SSep model on real data from AMI (individual headset)
> Speech separation models are normally trained on fully overlapped mixtures, often because it is not straightforward to train such models in a supervised way with real data.
In this experiment, a SSep model pretrained on LibriMix `sep_noisy` is finetuned using the recording from the individual headset of each speaker.
While the sources present some leakage (i.e. from the recording of speaker A it is possible to hear speaker B and C as well but at a lower energy), the finetuned models handle much better real data, drastically reducing phenomena like splitting a unique speaker in multiple outputs or aggregating multiple speakers in a unique output source.
2. `SSepVAD_train`: finetuning a SSep model + VAD for diarization, with the option of finetuning only the only the VAD or all the model E2E
> SSepVAD produces diarization results by applying VAD to each estimated source from a SSep model. Because the estimated sources are not perfect, the VAD needs to be finetuned. In this experiment, two strategies are explored: VAD-only finetuning and E2E.
3. `SSepVAD_inference`: performing online diarization inference on AMI using the online clustering framework provided by [Diart](https://github.com/juanmc2005/diart)
> The SSepVAD model is used as a custom model for the segmentation modelling part in Diart.

## SSep inference examples
The `example_inference_real_data.ipynb` notebook can be used to test the different models on real data from AMI.\
Some chosen examples with commentary can be found [here].

## Conda environments and dependencies
Each experiment folder has inside a `docker` folder which in turns contains an `environment.yml` file.\
The code relies on external codebases, most importantly `pyannote.audio` and `diart`, which are cloned inside the `externals` folder.
More info can be found in the `README.md` of the `externals` folder.

## Datasets
### SSep pretraining
The dataset used to pretrain the SSep model is [LibriMix](https://github.com/JorisCos/LibriMix), specifically the `sep_noisy` variant.

### SSep finetuning
The script `AMI_segments_x_spks_for_finetuning.py` performs random crops to train a separation model with 2 or 3 fixed outputs.
The code requires the [AMI corpus](https://groups.inf.ed.ac.uk/ami/corpus/) to be installed, as detailed [here](https://github.com/pyannote/AMI-diarization-setup)
It is best for the crops to always contain as many speakers as the number of outputs of the model.
As such, the duration of the crops depends on the number of speakers (and the data distribution of the AMI dataset).
For the dataset used to train the 2-outputs model, 5-second crops are used.
For the dataset used to train the 3-outputs model, 35-second crops are preferred, as there are very few crops of 5 seconds with 3 speakers in it (in AMI). 

### SSepVAD training and inference
The AMI datasets needs to be set up as in the `SSep finetuning` section, following the steps detailed [here](https://github.com/pyannote/AMI-diarization-setup).

## Useful resources
1. How to pretrain a SSep model on LibriMix?
    - A good choice is to use [Asteroid](https://github.com/asteroid-team/asteroid)
2. Where can I find pretrained models?
  - For SSep models pretrained on LibriMix, you can resort to the [Model Hub](https://huggingface.co/models?library=asteroid)
  - For SSep models finetuned on AMI and SSepVAD, stay tuned!...

## Citation
["ONLINE SPEAKER DIARIZATION OF MEETINGS GUIDED BY SPEECH SEPARATION"](https://hal.science/hal-04419041)
