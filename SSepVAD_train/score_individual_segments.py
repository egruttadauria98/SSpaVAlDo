"""
Evaluation script on the trained model on individual segments.
"""
import os
import torch

from speaker_diarization.inference.individual_segments.score import run_bootstrap_scoring
from speaker_diarization.models.pyannote_ss_vad_model import SS_VAD_model
from speaker_diarization.utils.print_utils import print_args_decorator

@print_args_decorator
def run_scoring(best_model_path,
                ss_model,
                vad_model,
                exp_dir,
                conf,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                ):

    print("\nDevice: ", device)

    # Load best checkpoint of the trained model
    model = SS_VAD_model.load_from_checkpoint(best_model_path, 
                                              ss_model=ss_model, 
                                              vad_model=vad_model, 
                                              map_location=torch.device(device))

    model.eval()

    df_scoring = run_bootstrap_scoring(
        model,
        dataset=conf["data"]["db_name"],
        protocol=conf["data"]["protocol_name"],
        metric_type=conf["eval"]["metric_type"],
        split=conf["eval"]["split"],
        segment_size=conf["eval"]["segment_size"],
        tau_list=conf["eval"]["tau_list"],
        n_bootstrap_samples=conf["eval"]["n_bootstrap_samples"],
        n_iterations_per_sample=conf["eval"]["n_iterations_per_sample"],
        max_spks=conf["eval"]["max_spks"], # Returns detailed DER on segments with only x speakers from 0 to max_spks
        oracle_spks_num=conf["eval"]["oracle_spks_num"],
        score_overlap_only=conf["eval"]["score_overlap_only"],
        device=device
    )

    df_scoring.to_csv(os.path.join(exp_dir, "scoring_individual_segments.csv"), index=False)
