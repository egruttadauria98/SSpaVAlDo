"""Wrapper of the Speaker Diarization task for more plotting options on TensorBoard."""
import math

import torch
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

from pyannote.audio.tasks import SpeakerDiarization
from pyannote.audio.utils.permutation import permutate

class SpeakerDiarizationEnhanced(SpeakerDiarization):
    """Wrapper of the Speaker Diarization task for more plotting options on TensorBoard.

    Enhancements to the original class:
    1. Plot every batch on TensorBoard, instead of when epoch index is a power of 2.
    2a. Plot the first 10 batches, instead of just the first batch.
    2b. Plot all elements in the batch, not just up to 9 (always 3 columns in the plot, then estimate ceil of rows).
    3. Only one time, load the audio in the elements of the first 10 batches given as input to the model.
    4. After the plots, for each batch, load the audio of the estimated sources as well.
    """
    def __init__(self, protocol, *args, **kwargs):
        super().__init__(protocol, *args, **kwargs)

        file = next(protocol.train())
        audio_path = str(file['audio'])
        _, self.sample_rate = sf.read(audio_path, 
                                      dtype='float32', 
                                      start=0, 
                                      stop=1)

    def validation_step(self, batch, batch_idx: int):
        """Compute validation loss and metric

        Parameters
        ----------
        batch : dict of torch.Tensor
            Current batch.
        batch_idx: int
            Batch index.
        """

        # target
        target = batch["y"]
        # (batch_size, num_frames, num_speakers)

        waveform = batch["X"]
        # (batch_size, num_channels, num_samples)

        # TODO: should we handle validation samples with too many speakers
        # waveform = waveform[keep]
        # target = target[keep]

        # forward pass
        prediction, normalized_sources = self.model(waveform, return_sources=True)
        batch_size, num_frames, _ = prediction.shape

        # frames weight
        weight_key = getattr(self, "weight", None)
        weight = batch.get(
            weight_key,
            torch.ones(batch_size, num_frames, 1, device=self.model.device),
        )
        # (batch_size, num_frames, 1)

        # warm-up
        warm_up_left = round(self.warm_up[0] / self.duration * num_frames)
        weight[:, :warm_up_left] = 0.0
        warm_up_right = round(self.warm_up[1] / self.duration * num_frames)
        weight[:, num_frames - warm_up_right :] = 0.0

        if self.specifications.powerset:
            multilabel = self.model.powerset.to_multilabel(prediction)
            permutated_target, _ = permutate(multilabel, target)

            # FIXME: handle case where target have too many speakers?
            # since we don't need
            permutated_target_powerset = self.model.powerset.to_powerset(
                permutated_target.float()
            )
            seg_loss = self.segmentation_loss(
                prediction, permutated_target_powerset, weight=weight
            )

        else:
            permutated_prediction, _ = permutate(target, prediction)
            seg_loss = self.segmentation_loss(
                permutated_prediction, target, weight=weight
            )

        self.model.log(
            "loss/val/segmentation",
            seg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if self.vad_loss is None:
            vad_loss = 0.0

        else:
            # TODO: vad_loss probably does not make sense in powerset mode
            # because first class (empty set of labels) does exactly this...
            if self.specifications.powerset:
                vad_loss = self.voice_activity_detection_loss(
                    prediction, permutated_target_powerset, weight=weight
                )

            else:
                vad_loss = self.voice_activity_detection_loss(
                    permutated_prediction, target, weight=weight
                )

            self.model.log(
                "loss/val/vad",
                vad_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        loss = seg_loss + vad_loss

        self.model.log(
            "loss/val",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if self.specifications.powerset:
            self.model.validation_metric(
                torch.transpose(
                    multilabel[:, warm_up_left : num_frames - warm_up_right], 1, 2
                ),
                torch.transpose(
                    target[:, warm_up_left : num_frames - warm_up_right], 1, 2
                ),
            )
        else:
            self.model.validation_metric(
                torch.transpose(
                    prediction[:, warm_up_left : num_frames - warm_up_right], 1, 2
                ),
                torch.transpose(
                    target[:, warm_up_left : num_frames - warm_up_right], 1, 2
                ),
            )

        self.model.log_dict(
            self.model.validation_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        #if (
        #    self.model.current_epoch == 0
        #    or math.log2(self.model.current_epoch) % 1 > 0
        #    or batch_idx > 0
        #):
        #    return
        if batch_idx >=10:
            return

        # Add input audio
        if self.model.current_epoch == 0:
            for i, audio in enumerate(waveform):
                for logger in self.model.loggers:
                    if isinstance(logger, TensorBoardLogger):
                        logger.experiment.add_audio(f"Batch_{batch_idx}_sample_{i}", 
                                                    audio.cpu().numpy().T, 
                                                    self.model.current_epoch, 
                                                    sample_rate=self.sample_rate)

        # visualize all validation samples of first 10 batches in Tensorboard/MLflow
        if self.specifications.powerset:
            y = permutated_target.float().cpu().numpy()
            y_pred = multilabel.cpu().numpy()
        else:
            y = target.float().cpu().numpy()
            y_pred = permutated_prediction.cpu().numpy()

        # prepare 3 x N grid (or smaller if batch size is smaller)
        #num_samples = min(self.batch_size, 9)
        num_samples = self.batch_size
        #nrows = math.ceil(math.sqrt(num_samples))
        #ncols = math.ceil(num_samples / nrows)
        ncols = 3
        nrows = math.ceil(num_samples / ncols)
        fig, axes = plt.subplots(
            nrows=2 * nrows, ncols=ncols, figsize=(8, 5), squeeze=False
        )

        # Add batch label text to the plots
        fig.suptitle(f"Batch_{batch_idx}", fontsize=16)

        # reshape target so that there is one line per class when plotting it
        y[y == 0] = np.NaN
        if len(y.shape) == 2:
            y = y[:, :, np.newaxis]
        y *= np.arange(y.shape[2])

        # plot each sample
        for sample_idx in range(num_samples):
            # find where in the grid it should be plotted
            row_idx = sample_idx // nrows
            col_idx = sample_idx % ncols

            # plot target
            ax_ref = axes[row_idx * 2 + 0, col_idx]
            sample_y = y[sample_idx]
            ax_ref.plot(sample_y)
            ax_ref.set_xlim(0, len(sample_y))
            ax_ref.set_ylim(-1, sample_y.shape[1])
            ax_ref.get_xaxis().set_visible(False)
            ax_ref.get_yaxis().set_visible(False)

            # plot predictions
            ax_hyp = axes[row_idx * 2 + 1, col_idx]
            sample_y_pred = y_pred[sample_idx]
            ax_hyp.axvspan(0, warm_up_left, color="k", alpha=0.5, lw=0)
            ax_hyp.axvspan(
                num_frames - warm_up_right, num_frames, color="k", alpha=0.5, lw=0
            )
            ax_hyp.plot(sample_y_pred)
            ax_hyp.set_ylim(-0.1, 1.1)
            ax_hyp.set_xlim(0, len(sample_y))
            ax_hyp.get_xaxis().set_visible(False)

        plt.tight_layout()

        for logger in self.model.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_figure(f"Batch_{batch_idx}_epoch_{self.model.current_epoch}", 
                                             fig, 
                                             self.model.current_epoch)
            elif isinstance(logger, MLFlowLogger):
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=fig,
                    artifact_file=f"samples_epoch{self.model.current_epoch}.png",
                )

        plt.close(fig)

        # Add estimated sources audio
        num_sources = len(normalized_sources)
        for b_idx, _ in enumerate(normalized_sources[0]):
            for s_idx in range(num_sources):
                for logger in self.model.loggers:
                    if isinstance(logger, TensorBoardLogger):
                        logger.experiment.add_audio(f"Batch_{batch_idx}_sample_{b_idx}_source_{s_idx}_epoch_{self.model.current_epoch}", 
                                                    normalized_sources[s_idx][b_idx].cpu().numpy().T, 
                                                    self.model.current_epoch, 
                                                    sample_rate=self.sample_rate) 