"""
Implements additional scheduler which inherits from pytorch_lightning.Callback
"""

import os
import math
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

def compute_total_number_steps(dataset_len, batch_size, max_epochs, n_gpus, batch_sizes=None, epochs=None):
  """Compute the total number of steps in the training.
  This function is needed for OneCycleLR and WeightDecayScheduler.
  """
  if batch_sizes is None:
    batch_sizes = []
  if epochs is None:
    epochs = []
  steps = 0
  batch_sizes = [batch_size] + batch_sizes
  epochs = [0] + epochs + [max_epochs]
  for i in range(len(batch_sizes)):
    steps += int((dataset_len /
                 (batch_sizes[i]*n_gpus)) * (epochs[i+1] - epochs[i]))
  return steps

class BatchSizeScheduler(pl.Callback):
  """Scheduler for the batch size, small batch size at the beginnig of
  the training that get bigger with the number of epochs.

  Parameters
  ----------
  batch_sizes : List[int]
    Size of each batch based on number of epochs.
  epochs : List[int]
    When to change the batch size.

  Attributes
  ----------
  batch_sizes : List[int]
    Size of each batch based on number of epochs.
  epochs : List[int]
    When to change the batch size.

  Methods
  -------
  on_train_epoch_end(trainer,pl_module)
    Reload the dataloader after epoch end.
  """

  def __init__(self, batch_sizes, epochs):
    super().__init__()
    self.batch_sizes = batch_sizes
    self.epochs = epochs

  def on_train_epoch_end(self, trainer, pl_module):
    """Reload the dataloader after epoch end
    """
    if trainer.current_epoch+1 in self.epochs:
      idx = self.epochs.index(trainer.current_epoch+1)
      batch_size = self.batch_sizes[idx]
      train_dataloader = DataLoader(pl_module.train_dataloader().dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=pl_module.train_dataloader().num_workers,
                                    drop_last=pl_module.train_dataloader().drop_last)
      val_dataloader = DataLoader(pl_module.val_dataloader().dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=pl_module.val_dataloader().num_workers,
                                  drop_last=pl_module.val_dataloader().drop_last)

      # Replace the existing DataLoader objects in the trainer with the new ones
      pl_module.train_loader = train_dataloader
      pl_module.val_loader = val_dataloader


class WeightDecayScheduler(pl.Callback):
  """Scheduler for the weight decay, Cosine increasing with epochs.

  Parameters
  ----------
  wd_min : float
    Starting point of the weight decay.
  wd_max : float
    Ending point of the weight decay.
  total_steps : int
    Total number of steps in the training.

  Attributes
  ----------
  wd_min : float
    Starting point of the weight decay.
  wd_max : float
    Ending point of the weight decay.
  total_steps : int
    Total number of steps in the training.

  Methods
  -------
  on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
    After each batch changes weight decay.
  """

  def __init__(self, wd_min, wd_max, total_steps):
    super().__init__()
    self.wd_min = wd_min
    self.wd_max = wd_max
    self.total_steps = total_steps

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    """After each batch changes weight decay"""
    current_step = trainer.global_step
    wd = self.wd_max + 0.5 * \
        (1 + math.cos(math.pi * current_step / self.total_steps)) * \
        (self.wd_min - self.wd_max)
    optimizer = trainer.optimizers[0]
    for group in optimizer.param_groups:
      group["weight_decay"] = wd
