from .PreProcessImage import PreProcessIm
from .Prefetcher import Prefetcher
import numpy as np


class Dataset(object):
  """The core elements of a dataset.    
  Args:
    final_batch: bool. The last batch may not be complete, if to abandon this 
      batch, set 'final_batch' to False.
  """

  def __init__(
      self,
      dataset_size=None,
      batch_size=None,
      final_batch=True,
      shuffle=True,
      num_prefetch_threads=1,
      prng=np.random,
      **pre_process_im_kwargs):

    self.pre_process_im = PreProcessIm(
      prng=prng,
      **pre_process_im_kwargs)

    self.prefetcher = Prefetcher(
      self.get_sample,
      dataset_size,
      batch_size,
      final_batch=final_batch,
      num_threads=num_prefetch_threads)

    self.shuffle = shuffle
    self.epoch_done = True
    self.prng = prng

  def set_mirror_type(self, mirror_type):
    self.pre_process_im.set_mirror_type(mirror_type)

  def get_sample(self, ptr):
    """Get one sample to put to queue."""
    raise NotImplementedError

  def next_batch(self):
    """Get a batch from the queue."""
    raise NotImplementedError

  def set_batch_size(self, batch_size):
    """You can change batch size, had better at the beginning of a new epoch.
    """
    self.prefetcher.set_batch_size(batch_size)
    self.epoch_done = True

  def stop_prefetching_threads(self):
    """This can be called to stop threads, e.g. after finishing using the 
    dataset, or when existing the python main program."""
    self.prefetcher.stop()
