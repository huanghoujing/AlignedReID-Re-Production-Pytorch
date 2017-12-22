from aligned_reid.utils.dataset_utils import PreProcessIm
from aligned_reid.utils.dataset_utils import Prefetcher
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
      resize_size=None,
      crop_size=None,
      scale=True,
      im_mean=None,
      im_std=None,
      mirror_type=None,
      batch_dims='NCHW',
      num_prefetch_threads=1,
      prng=np.random):

    self.pre_process_im = PreProcessIm(
      resize_size=resize_size,
      crop_size=crop_size,
      scale=scale,
      im_mean=im_mean,
      im_std=im_std,
      mirror_type=mirror_type,
      batch_dims=batch_dims,
      prng=prng)

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
