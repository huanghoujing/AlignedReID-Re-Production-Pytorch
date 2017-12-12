from __future__ import print_function
import os
import os.path as osp
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import threading
import Queue
import time
from collections import defaultdict
import shutil

new_im_name_tmpl = '{:08d}_{:04d}_{:08d}.jpg'

def parse_im_name(im_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    parsed = int(im_name[:8])
  else:
    parsed = int(im_name[9:13])
  return parsed


def get_im_names(im_dir, pattern='*.jpg', return_np=True, return_path=False):
  """Get the image names in a dir. Optional to return numpy array, paths."""
  im_paths = glob.glob(osp.join(im_dir, pattern))
  im_names = [osp.basename(path) for path in im_paths]
  ret = im_paths if return_path else im_names
  if return_np:
    ret = np.array(ret)
  return ret


def move_ims(ori_im_paths, new_im_dir, parse_im_name, new_im_name_tmpl):
  """Rename and move images to new directory."""
  cnt = defaultdict(int)
  new_im_names = []
  for im_path in ori_im_paths:
    im_name = osp.basename(im_path)
    id = parse_im_name(im_name, 'id')
    cam = parse_im_name(im_name, 'cam')
    cnt[(id, cam)] += 1
    new_im_name = new_im_name_tmpl.format(id, cam, cnt[(id, cam)] - 1)
    shutil.copy(im_path, osp.join(new_im_dir, new_im_name))
    new_im_names.append(new_im_name)
  return new_im_names


class PreProcessIm(object):
  def __init__(self, resize_size=None, crop_size=None,
               scale=True, im_mean=None, im_std=None,
               mirror_type=None, batch_dims='NCHW',
               prng=np.random):
    """
    Args:
      resize_size: (width, height) after resizing. If `None`, no resizing.
      crop_size: (width, height) after cropping. If `None`, no cropping.
      scale: whether to scale the pixel value by 1/255
      batch_dims: either 'NCHW' or 'NHWC'. 'N': batch size, 'C': num channels, 
        'H': im height, 'W': im width. PyTorch uses 'NCHW', while TensorFlow 
        uses 'NHWC'.
      prng: can be set to a numpy.random.RandomState object, in order to have 
        random seed independent from the global one
    """
    self.resize_size = resize_size
    self.crop_size = crop_size
    self.scale = scale
    self.im_mean = im_mean
    self.im_std = im_std
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type
    self.check_batch_dims(batch_dims)
    self.batch_dims = batch_dims
    self.prng = prng

  def __call__(self, im):
    return self.pre_process_im(im)

  @staticmethod
  def check_mirror_type(mirror_type):
    assert mirror_type in [None, 'random', 'always']

  @staticmethod
  def check_batch_dims(batch_dims):
    # 'N': batch size, 'C': num channels, 'H': im height, 'W': im width
    # PyTorch uses 'NCHW', while TensorFlow uses 'NHWC'.
    assert batch_dims in ['NCHW', 'NHWC']

  def set_mirror_type(self, mirror_type):
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type

  @staticmethod
  def rand_crop_im(im, new_size, prng=np.random):
    """Crop `im` to `new_size`: [new_w, new_h]."""
    if (new_size[0] == im.shape[1]) and (new_size[1] == im.shape[0]):
      return im
    h_start = prng.randint(0, im.shape[0] - new_size[1])
    w_start = prng.randint(0, im.shape[1] - new_size[0])
    im = np.copy(
      im[h_start: h_start + new_size[1], w_start: w_start + new_size[0], :])
    return im

  def pre_process_im(self, im):
    """Pre-process image.
    `im` is a numpy array returned by matplotlib.pyplot.imread()."""
    # Resize.
    if self.resize_size is not None:
      im = cv2.resize(im, self.resize_size, interpolation=cv2.INTER_LINEAR)
    # Randomly crop a sub-image.
    if self.crop_size is not None:
      im = self.rand_crop_im(im, self.crop_size, prng=self.prng)
    # scaled by 1/255.
    if self.scale:
      im = im / 255.
    # Subtract mean and scaled by std
    # im -= np.array(self.im_mean) # This causes an error:
    # Cannot cast ufunc subtract output from dtype('float64') to
    # dtype('uint8') with casting rule 'same_kind'
    if self.im_mean is not None:
      im = im - np.array(self.im_mean)
    if self.im_mean is not None and self.im_std is not None:
      im = im / np.array(self.im_std).astype(float)
    # May mirror image.
    mirrored = False
    if self.mirror_type == 'always' \
        or (self.mirror_type == 'random' and self.prng.uniform() > 0.5):
      im = im[:, ::-1, :]
      mirrored = True
    # The original image has dims 'HWC', transform it to 'CHW'.
    if self.batch_dims == 'NCHW':
      im = im.transpose(2, 0, 1)

    return im, mirrored


def calculate_im_mean(im_dir=None, pattern='*.jpg', im_paths=None,
                       mean_file=None):
  """Calculate the mean values of R, G and B.
  Args:
    im_dir: a dir containing images. If `im_paths` is provided, this is not 
    used.
    pattern: the file pattern for glob.glob()
    im_paths: a list of image paths
    mean_file: a file to save image mean. If None, results will not be saved.
  Returns:
    A numpy array with shape [3], for R, G, B mean value respectively.
  """

  # Get image paths.
  if im_paths is None:
    im_names = get_im_names(im_dir, pattern=pattern)
    im_paths = [osp.join(im_dir, name) for name in im_names]

  # Calculate mean.
  num_pixels = []
  values_sum = np.zeros([3])
  for path in im_paths:
    im = plt.imread(path)
    num_pixels.append(im.shape[0] * im.shape[1])
    values_sum += np.sum(np.sum(im, axis=0), axis=0)
  im_mean = values_sum / np.sum(num_pixels)

  # Write the mean to file.
  if mean_file is not None:
    mean_dir = osp.dirname(mean_file)
    if not osp.exists(mean_dir):
      os.makedirs(mean_dir)
    with open(mean_file, 'w') as f:
      pickle.dump(im_mean, f)
      print('Saved image mean to file {}.'.format(mean_file))

  return im_mean


def partition_train_val_set(im_names, parse_im_name,
                            num_val_ids=None, val_prop=None, seed=1):
  """Partition the trainval set into train and val set. 
  Args:
    im_names: trainval image names
    parse_im_name: a function to parse id and camera from image name
    num_val_ids: number of ids for val set. If not set, val_prob is used.
    val_prop: the proportion of validation ids
    seed: the random seed to reproduce the partition results. If not to use, 
      then set to `None`.
  Returns:
    a dict with keys (`train_im_names`, 
                      `val_query_im_names`, 
                      `val_gallery_im_names`)
  """
  np.random.seed(seed)
  # Transform to numpy array for slicing.
  if not isinstance(im_names, np.ndarray):
    im_names = np.array(im_names)
  np.random.shuffle(im_names)
  ids = np.array([parse_im_name(n, 'id') for n in im_names])
  cams = np.array([parse_im_name(n, 'cam') for n in im_names])
  unique_ids = np.unique(ids)
  np.random.shuffle(unique_ids)

  # Query indices and gallery indices
  query_inds = []
  gallery_inds = []

  if num_val_ids is None:
    assert 0 < val_prop < 1
    num_val_ids = int(len(unique_ids) * val_prop)
  num_selected_ids = 0
  for unique_id in unique_ids:
    query_inds_ = []
    # The indices of this id in trainval set.
    inds = np.argwhere(unique_id == ids).flatten()
    # The cams that this id has.
    unique_cams = np.unique(cams[inds])
    # For each cam, select one image for query set.
    for unique_cam in unique_cams:
      query_inds_.append(
        inds[np.argwhere(cams[inds] == unique_cam).flatten()[0]])
    gallery_inds_ = list(set(inds) - set(query_inds_))
    # For each query image, if there is no same-id different-cam images in
    # gallery, put it in gallery.
    for query_ind in query_inds_:
      if len(gallery_inds_) == 0 \
          or len(np.argwhere(cams[gallery_inds_] != cams[query_ind])
                     .flatten()) == 0:
        query_inds_.remove(query_ind)
        gallery_inds_.append(query_ind)
    # If no query image is left, leave this id in train set.
    if len(query_inds_) == 0:
      continue
    query_inds.append(query_inds_)
    gallery_inds.append(gallery_inds_)
    num_selected_ids += 1
    if num_selected_ids >= num_val_ids:
      break

  query_inds = np.hstack(query_inds)
  gallery_inds = np.hstack(gallery_inds)
  val_inds = np.hstack([query_inds, gallery_inds])
  trainval_inds = np.arange(len(im_names))
  train_inds = np.setdiff1d(trainval_inds, val_inds)

  train_inds = np.sort(train_inds)
  query_inds = np.sort(query_inds)
  gallery_inds = np.sort(gallery_inds)

  partitions = dict(train_im_names=im_names[train_inds],
                    val_query_im_names=im_names[query_inds],
                    val_gallery_im_names=im_names[gallery_inds])

  return partitions


class Counter(object):
  """A thread safe counter."""

  def __init__(self, val=0, max_val=0):
    self._value = val
    self.max_value = max_val
    self._lock = threading.Lock()

  def reset(self):
    with self._lock:
      self._value = 0

  def set_max_value(self, max_val):
    self.max_value = max_val

  def increment(self):
    with self._lock:
      if self._value < self.max_value:
        self._value += 1
        incremented = True
      else:
        incremented = False
      return incremented, self._value

  def get_value(self):
    with self._lock:
      return self._value


class Enqueuer(object):
  def __init__(self, get_element, num_elements, num_threads=1, queue_size=20):
    """
    Args:
      get_element: a function that takes a pointer and returns an element
      num_elements: total number of elements to put into the queue
      num_threads: num of parallel threads, >= 1
      queue_size: the maximum size of the queue. Set to some positive integer 
        to save memory, otherwise, set to 0. 
    """
    self.get_element = get_element
    assert num_threads > 0
    self.num_threads = num_threads
    self.queue_size = queue_size
    self.queue = Queue.Queue(maxsize=queue_size)
    # The pointer shared by threads.
    self.ptr = Counter(max_val=num_elements)
    # The event to wake up threads, it's set at the beginning of an epoch.
    # It's cleared after an epoch is enqueued or when the states are reset.
    self.event = threading.Event()
    # To reset states.
    self.reset_event = threading.Event()
    # The event to terminate the threads.
    self.stop_event = threading.Event()
    self.threads = []
    for _ in range(num_threads):
      thread = threading.Thread(target=self.enqueue)
      # Set the thread in daemon mode, so that the main program ends normally.
      thread.daemon = True
      thread.start()
      self.threads.append(thread)

  def start_ep(self):
    """Start enqueuing an epoch."""
    self.event.set()

  def end_ep(self):
    """When all elements are enqueued, let threads sleep to save resources."""
    self.event.clear()
    self.ptr.reset()

  def reset(self):
    """Reset the threads, pointer and the queue to initial states. In common 
    case, this will not be called."""
    self.reset_event.set()
    self.event.clear()
    # wait for threads to pause. This is not an absolutely safe way. The safer
    # way is to check some flag inside a thread, not implemented yet.
    time.sleep(5)
    self.reset_event.clear()
    self.ptr.reset()
    self.queue = Queue.Queue(maxsize=self.queue_size)

  def set_num_elements(self, num_elements):
    """Reset the max number of elements."""
    self.reset()
    self.ptr.set_max_value(num_elements)

  def stop(self):
    """Wait for threads to terminate."""
    self.stop_event.set()
    for thread in self.threads:
      thread.join()

  def enqueue(self):
    while not self.stop_event.isSet():
      # If the enqueuing event is not set, the thread just waits.
      if not self.event.wait(0.5): continue
      # Increment the counter to claim that this element has been enqueued by
      # this thread.
      incremented, ptr = self.ptr.increment()
      if incremented:
        element = self.get_element(ptr - 1)
        # When enqueuing, keep an eye on the stop and reset signal.
        while not self.stop_event.isSet() and not self.reset_event.isSet():
          try:
            # This operation will wait at most `timeout` for a free slot in
            # the queue to be available.
            self.queue.put(element, timeout=0.5)
            break
          except:
            pass
      else:
        self.end_ep()
    print('Exiting thread {}!!!!!!!!'.format(threading.current_thread().name))


class Prefetcher(object):
  """This helper class enables sample enqueuing and batch dequeuing, to speed 
  up batch fetching. It abstracts away the enqueuing and dequeuing logic."""

  def __init__(self, get_sample, dataset_size, batch_size, final_batch=True,
               num_threads=1, prefetch_size=200):
    """
    Args:
      get_sample: a function that takes a pointer (index) and returns a sample
      dataset_size: total number of samples in the dataset
      final_batch: True or False, whether to keep or drop the final incomplete 
        batch
      num_threads: num of parallel threads, >= 1
      prefetch_size: the maximum size of the queue. Set to some positive integer 
        to save memory, otherwise, set to 0.
    """
    self.full_dataset_size = dataset_size
    self.final_batch = final_batch
    final_sz = self.full_dataset_size % batch_size
    if not final_batch:
      dataset_size = self.full_dataset_size - final_sz
    self.dataset_size = dataset_size
    self.batch_size = batch_size
    self.enqueuer = Enqueuer(get_element=get_sample, num_elements=dataset_size,
                             num_threads=num_threads, queue_size=prefetch_size)
    # The pointer indicating whether an epoch has been fetched from the queue
    self.ptr = 0
    self.ep_done = True

  def set_batch_size(self, batch_size):
    """You had better change batch size at the beginning of a new epoch."""
    final_sz = self.full_dataset_size % batch_size
    if not self.final_batch:
      self.dataset_size = self.full_dataset_size - final_sz
    self.enqueuer.set_num_elements(self.dataset_size)
    self.batch_size = batch_size
    self.ep_done = True

  def next_batch(self):
    """Return a batch of samples, meanwhile indicate whether the epoch is 
    done. The purpose of this func is mainly to abstract away the loop and the
    boundary-checking logic.
    Returns:
      samples: a list of samples
      done: bool, whether the epoch is done
    """
    # Start enqueuing and other preparation at the beginning of an epoch.
    if self.ep_done:
      self.start_ep_prefetching()
    # Whether an epoch is done.
    self.ep_done = False
    samples = []
    for _ in range(self.batch_size):
      # Indeed, `>` will not occur.
      if self.ptr >= self.dataset_size:
        self.ep_done = True
        break
      else:
        self.ptr += 1
        sample = self.enqueuer.queue.get()
        # print('queue size {}'.format(self.enqueuer.queue.qsize()))
        samples.append(sample)
    # print 'queue size: {}'.format(self.enqueuer.queue.qsize())
    # Indeed, `>` will not occur.
    if self.ptr >= self.dataset_size:
      self.ep_done = True
    return samples, self.ep_done

  def start_ep_prefetching(self):
    """
    NOTE: Has to be called at the start of every epoch.
    """
    self.enqueuer.start_ep()
    self.ptr = 0

  def stop(self):
    """This can be called to stop threads, e.g. after finishing using the 
    dataset, or when existing the python main program."""
    self.enqueuer.stop()
