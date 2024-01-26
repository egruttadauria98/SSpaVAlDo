"""
Evaluation function on individual short segments of a dataset.
Author: Elio Gruttadauria
"""

import torch
import numpy as np
import pandas as pd
from scipy import stats
import soundfile as sf
import torchaudio

from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.detection import DetectionErrorRate
from speaker_diarization.data.pyannote_data_protocol import PyannoteDataProtocol

def calculate_confidence_interval(data, confidence=0.95):
    # Calculate the mean
    mean = np.mean(data)
    # Calculate the standard error of the mean
    sem = stats.sem(data)
    # Calculate the margin of error
    moe = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    # Calculate the lower and upper bounds
    lower_bound = mean - moe
    upper_bound = mean + moe
    return [mean, moe, [lower_bound, upper_bound]]

def load_input_waveform(path, resample, start=None, stop=None):
    # Return tensor with shape (batch, channels, samples)
    sig, sr = sf.read(path, dtype="float32", start=start, stop=stop)
    sig = torch.from_numpy(sig)
    if resample and sr != resample:
        sig = torchaudio.functional.resample(
                sig, sr, resample
            )
    if len(sig.shape) > 2:
        raise NotImplementedError
    elif len(sig.shape) == 2:
        # Assuming samples dimension bigger than channel one
        if sig.shape[0] > sig.shape[1]:
            sig = sig.T
        sig = sig.unsqueeze(0)
    else:    
        sig = sig.unsqueeze(0).unsqueeze(0)
    return sig


def run_bootstrap_scoring(
    model,
    dataset,
    protocol,
    metric_type="DER",
    split="test",
    segment_size=5,
    tau_list=[0.45, 0.5, 0.55],
    n_bootstrap_samples=10,
    n_iterations_per_sample=25,
    max_spks=5,
    oracle_spks_num=False,
    score_overlap_only=False,
    resample=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):

    """Given a model, run inference on segments of a few seconds for evaluation.
    The segments are cropped randomly from a real test set of longer recordings.
    As such, cropping works as a sampling operation from the real distribution of
    the dataset: a bootstrap.
    The code returns both points estimates and error measure.

    Run bootstrap scoring for speaker diarization evaluation.

    Parameters
    ----------
    model : object
        The torch model to be used for scoring.
    dataset : string
        Name of the dataset, as configured in the Pyannote Data Protocol
        ex. "AMI"
    protcol : string
        Name of the protocol, as configured in the Pyannote Data Protocol
        ex. "AMI.SpeakerDiarization.only_words"
    metric_type : string, optional
        The metric to be used for evaluation (default is 'DER').
        Currently supporting only DER and VAD.
    split : str, optional
        The dataset split to be used (default is 'test').
    segment_size : int, optional
        Size of each segment for scoring in minutes (default is 5).
    tau_list : list of float, optional
        A list of threshold values to be used for evaluation.
        The threshold are used to convert continuous prediction 
        into discrete binary values.
    n_bootstrap_samples : int, optional
        Number of bootstrap samples to generate (default is 10).
    n_iterations_per_sample : int, optional
        Number of iterations per bootstrap sample (default is 25).
    max_spks : int, optional
        Maximum number of speakers for detailed DER calculation (default is 5).
    oracle_spks_num : bool, optional
        Whether to use oracle number of speakers (default is False).
    score_overlap_only : bool, optional
        If set to True, scoring is done only on overlapping speech segments (default is False).
    resample : bool or int, optional
        Whether to resample the audio (default is False).
        If not False, it must be an integer representing the new sample rate.
    device : torch.device, optional
        The device (CPU or CUDA) on which to run the model (default depends on CUDA availability).

    Returns
    -------
    df_scoring : pd.DataFrame
        A dataframe with all the evaluation metrics
    """

    assert split in ["train", "dev", "test"]

    print("\nDevice: ", device)

    # Load data
    pyannote_data = PyannoteDataProtocol(db_name=dataset)
    pyannote_data.load_protocol(protocol_name=protocol)
    set_iterator = pyannote_data.get_split_iterator(split=split)

    # Define dictionary to collect all data to return
    dict_bootstrap = dict()
    dict_bootstrap['tau_active'] = []
    dict_bootstrap['der_abs'] = []
    dict_bootstrap['der_perc'] = []
    dict_bootstrap['fa_perc'] = []
    dict_bootstrap['ms_perc'] = []
    dict_bootstrap['sc_perc'] = []
    dict_bootstrap['n_bootstrap_samples'] = []
    dict_bootstrap['n_iterations_per_sample'] = []

    #dict_bootstrap['der_0spks'] = []
    #dict_bootstrap['der_1spks'] = []
    #dict_bootstrap['der_2spks'] = []
    #dict_bootstrap['der_3spks'] = []
    #dict_bootstrap['der_4spks'] = []
    #dict_bootstrap['der_5spks'] = []

    for spk_idx in range(max_spks+1):
        dict_bootstrap[f'der_{spk_idx}spks'] = []

    # Define DER metrics
    if metric_type == "DER":
        metric = DiarizationErrorRate()
    elif metric_type == "VAD":
        metric = DetectionErrorRate()
    else:
        raise NotImplementedError

    #spks0_der = DiarizationErrorRate()
    #spks1_der = DiarizationErrorRate()
    #spks2_der = DiarizationErrorRate()
    #spks3_der = DiarizationErrorRate()
    #spks4_der = DiarizationErrorRate()
    #spks5_der = DiarizationErrorRate()

    # This list contains DER scoring object
    # It needs to be hashed with the relative number of speakers value
    # Ex. spk_der_list[0] -> DER object for segments with 0 speakers
    spk_der_list = []
    for _ in range(max_spks+1):
        if metric_type == "DER":
            spk_der_list.append(DiarizationErrorRate())
        elif metric_type == "VAD":
            spk_der_list.append(DetectionErrorRate())
        else:
            raise NotImplementedError

    for tau_active in tau_list:
        print(f"\nWorking on tau active: {tau_active}")

        dict_bootstrap['tau_active'].append(tau_active)
        dict_bootstrap['n_bootstrap_samples'].append(n_bootstrap_samples)
        dict_bootstrap['n_iterations_per_sample'].append(n_iterations_per_sample)
        dict_bootstrap['der_abs'].append([])
        dict_bootstrap['der_perc'].append([])
        dict_bootstrap['fa_perc'].append([])
        dict_bootstrap['ms_perc'].append([])
        dict_bootstrap['sc_perc'].append([])
        
        #dict_bootstrap['der_0spks'].append([])
        #dict_bootstrap['der_1spks'].append([])
        #dict_bootstrap['der_2spks'].append([])
        #dict_bootstrap['der_3spks'].append([])
        #dict_bootstrap['der_4spks'].append([])
        #dict_bootstrap['der_5spks'].append([])

        for spk_idx in range(max_spks+1):
            dict_bootstrap[f'der_{spk_idx}spks'].append([])

        for bootstrap_iter in range(n_bootstrap_samples):
            print(f"\n\tWorking on bootstrap sample: {bootstrap_iter+1} out of {n_bootstrap_samples}")

            metric.reset()

            #spks0_der.reset()
            #spks1_der.reset()
            #spks2_der.reset()
            #spks3_der.reset()
            #spks4_der.reset()
            #spks5_der.reset()

            for spk_idx in range(max_spks+1):
                spk_der_list[spk_idx].reset()

            iter_time_list = []

            for sample_iter in range(n_iterations_per_sample):
                for file in set_iterator():

                    # Make initial crop
                    (annotation_crop, 
                    info_crop) = PyannoteDataProtocol.generate_random_crop_file(file, 
                                                                                segment_size)

                    if oracle_spks_num and score_overlap_only:
                        a = 0
                        num_spks_crop = PyannoteDataProtocol.get_num_speakers(annotation_crop)
                        len_overlap = len(PyannoteDataProtocol.get_uem(annotation_crop, overlap_only=True))

                        # If 
                        #   the number of speakers is not the oracle one, OR 
                        #   there is no overlap, AND
                        #   the iteration counter is below 500
                        # -> random crop again
                        while (num_spks_crop != oracle_spks_num or len_overlap==0) and a<500:
                            (annotation_crop, 
                            info_crop) = PyannoteDataProtocol.generate_random_crop_file(file, 
                                                                                        segment_size)
                            a += 1

                    else:
                        if oracle_spks_num:
                            a = 0
                            # Continue until get a segment with oracle number of speakers
                            num_spks_crop = PyannoteDataProtocol.get_num_speakers(annotation_crop)
                            while (num_spks_crop != oracle_spks_num) and a<100:
                                (annotation_crop, 
                                info_crop) = PyannoteDataProtocol.generate_random_crop_file(file, 
                                                                                            segment_size)
                                a += 1

                        elif score_overlap_only:
                            a = 0
                            # Continue until get a segment with an overlap
                            while len(annotation_crop.get_overlap())==0 and a<100:
                                (annotation_crop, 
                                info_crop) = PyannoteDataProtocol.generate_random_crop_file(file, 
                                                                                            segment_size)
                                a += 1

                    n_spks = PyannoteDataProtocol.get_num_speakers(annotation_crop)

                    waveform = load_input_waveform(path=file['audio'], 
                                                   resample=resample,
                                                   start=info_crop['start_sample'], 
                                                   stop=info_crop['stop_sample']).to(device)
                    
                    import time
                    start_time = time.perf_counter()

                    est_activities = model(waveform)
                    
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    iter_time_list.append(elapsed_time)
                    #print(f"Execution time: {elapsed_time} seconds")
                    
                    est_activities = est_activities.squeeze(0).detach().cpu().numpy()

                    out, output_window = PyannoteDataProtocol.binarize_activities(est_activities=est_activities, 
                                                                                  segment_size=segment_size,
                                                                                  start_sec=info_crop['start_sec'],
                                                                                  tau_active=tau_active,
                                                                                  return_continuous=True)

                    #save_examples[file['uri']+f"_{sample_iter}"] = [[info_crop['start_sec'], info_crop['stop_sec']], 
                    #                                    annotation_crop, 
                    #                                    out, 
                    #                                    output_window, 
                    #                                    wave.cpu()]


                    #if score_overlap_only:
                    #    uem = annotation_crop.get_overlap()
                    #else:
                    #    uem = annotation_crop.get_timeline().extent()

                    uem = PyannoteDataProtocol.get_uem(annotation_crop, 
                                                       overlap_only=score_overlap_only)

                    # Score
                    _ = metric(
                        annotation_crop, # reference annotation
                        out,             # hypothesized annotation
                        uem=uem,         # scoring region
                        detailed=True
                    )

                    #if n_spks == 0:
                    #    _ = spks0_der(annotation_crop, out, uem=uem, detailed=True)
                    #elif n_spks == 1:
                    #    _ = spks1_der(annotation_crop, out, uem=uem, detailed=True)
                    #elif n_spks == 2:
                    #    _ = spks2_der(annotation_crop, out, uem=uem, detailed=True)
                    #elif n_spks == 3:
                    #    _ = spks3_der(annotation_crop, out, uem=uem, detailed=True)
                    #elif n_spks == 4:
                    #    _ = spks4_der(annotation_crop, out, uem=uem, detailed=True)
                    #elif n_spks == 5:
                    #    _ = spks5_der(annotation_crop, out, uem=uem, detailed=True)
                    #else:
                    #    raise("Not possible")

                    try:
                        _ = spk_der_list[n_spks](annotation_crop, out, uem=uem, detailed=True)
                    except IndexError:
                        raise("Found a segment with more than `max_spks`")

            # Aggregate the performance over the whole test set
            der_abs = abs(metric)
            print(f'\nDetection error rate = {der_abs * 100:.1f}% with tau_active = {tau_active}')

            last_row_report = metric.report().iloc[-1]
            print(last_row_report)

            #print(f'\n\tDER @ 0 spks = {abs(spks0_der) * 100:.1f}% with tau_active = {tau_active}')
            #print(f'\n\tDER @ 1 spks = {abs(spks1_der) * 100:.1f}% with tau_active = {tau_active}')
            #print(f'\n\tDER @ 2 spks = {abs(spks2_der) * 100:.1f}% with tau_active = {tau_active}')
            #print(f'\n\tDER @ 3 spks = {abs(spks3_der) * 100:.1f}% with tau_active = {tau_active}')
            #print(f'\n\tDER @ 4 spks = {abs(spks4_der) * 100:.1f}% with tau_active = {tau_active}')
            #print(f'\n\tDER @ 5 spks = {abs(spks5_der) * 100:.1f}% with tau_active = {tau_active}\n')
            
            for spk_idx in range(max_spks+1):
                print(f'\n\tDER @ {spk_idx} spks = {abs(spk_der_list[spk_idx]) * 100:.1f}% with tau_active = {tau_active}')


            # Get the last element of the dict, which will be a list
            # Inside the list, add the value of the current bootstrap sample
            dict_bootstrap['der_abs'][-1].append(der_abs)

            if metric_type == "DER":
              dict_bootstrap['der_perc'][-1].append(last_row_report['diarization error rate']['%'])
              dict_bootstrap['fa_perc'][-1].append(last_row_report['false alarm']['%'])
              dict_bootstrap['ms_perc'][-1].append(last_row_report['missed detection']['%'])
              dict_bootstrap['sc_perc'][-1].append(last_row_report['confusion']['%'])

            else:
              dict_bootstrap['der_perc'][-1].append(last_row_report['detection error rate']['%'])
              dict_bootstrap['fa_perc'][-1].append(last_row_report['false alarm']['%'])
              dict_bootstrap['ms_perc'][-1].append(last_row_report['miss']['%'])
              dict_bootstrap['sc_perc'][-1].append(0)

            #dict_bootstrap['der_0spks'][-1].append(abs(spks0_der))
            #dict_bootstrap['der_1spks'][-1].append(abs(spks1_der))
            #dict_bootstrap['der_2spks'][-1].append(abs(spks2_der))
            #dict_bootstrap['der_3spks'][-1].append(abs(spks3_der))
            #dict_bootstrap['der_4spks'][-1].append(abs(spks4_der))
            #dict_bootstrap['der_5spks'][-1].append(abs(spks5_der))

            for spk_idx in range(max_spks+1):
                dict_bootstrap[f'der_{spk_idx}spks'][-1].append(abs(spk_der_list[spk_idx]))

            time_metrics = calculate_confidence_interval(iter_time_list)
            print("\nExecution time metrics")
            print(f"Average execution time: {time_metrics[0]}")
            print(f"Margin of error: {time_metrics[1]}")
            print(f"Confidence interval: {time_metrics[2]}")

    df_scoring = pd.DataFrame(dict_bootstrap)

    for col in df_scoring.columns:
        if col not in ['tau_active', 'n_bootstrap_samples', 'n_iterations_per_sample']:
            df_scoring[col+"_error"] = df_scoring[col].apply(calculate_confidence_interval)

    return df_scoring








