from __future__ import print_function
import sys
import os
import os.path as osp
import cPickle as pickle
import gc
import numpy as np
from scipy import io
import datetime

import torch
from torch.autograd import Variable


def time_str(fmt=None):
  if fmt is None:
    fmt = '%Y-%m-%d_%H:%M:%S'
  return datetime.datetime.today().strftime(fmt)


def load_pickle(path):
  """Check and load pickle object.
  According to this post: https://stackoverflow.com/a/41733927, cPickle and 
  disabling garbage collector helps with loading speed."""
  assert osp.exists(path)
  # gc.disable()
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  # gc.enable()
  return ret


def save_pickle(obj, path):
  """Create dir and save file."""
  may_make_dir(osp.dirname(osp.abspath(path)))
  with open(path, 'wb') as f:
    pickle.dump(obj, f, protocol=2)


def load_hickle(path):
  """Check and load hickle object.
  hickle uses HDF5 protocol and is really FAST! 
  https://github.com/telegraphic/hickle"""
  assert osp.exists(path)
  import hickle
  ret = hickle.load(path)
  return ret


def save_hickle(obj, path):
  """Create dir and save hickle object.
  hickle uses HDF5 protocol and is really FAST! 
  https://github.com/telegraphic/hickle
  Typical extension of file is `hkl`.
  """
  import hickle
  may_make_dir(osp.dirname(osp.abspath(path)))
  hickle.dump(obj, path)


def save_mat(ndarray, path):
  """Save a numpy ndarray as .mat file."""
  io.savemat(path, dict(ndarray=ndarray))


def to_scalar(vt):
  """Transform a length-1 pytorch Variable or Tensor to scalar. 
  Suppose tx is a torch Tensor with shape tx.size() = torch.Size([1]), 
  then npx = tx.cpu().numpy() has shape (1,), not 1."""
  if isinstance(vt, Variable):
    return vt.data.cpu().numpy().flatten()[0]
  if torch.is_tensor(vt):
    return vt.cpu().numpy().flatten()[0]
  raise TypeError('Input should be a variable or tensor')


def transfer_optim_state(state, device_id=-1):
  """Transfer an optimizer.state to cpu or specified gpu, which means 
  transferring tensors of the optimizer.state to specified device. 
  The modification is in place for the state.
  Args:
    state: An torch.optim.Optimizer.state
    device_id: gpu id, or -1 which means transferring to cpu
  """
  for key, val in state.items():
    if isinstance(val, dict):
      transfer_optim_state(val, device_id=device_id)
    elif isinstance(val, Variable):
      raise RuntimeError("Oops, state[{}] is a Variable!".format(key))
    elif isinstance(val, torch.nn.Parameter):
      raise RuntimeError("Oops, state[{}] is a Parameter!".format(key))
    else:
      try:
        if device_id == -1:
          state[key] = val.cpu()
        else:
          state[key] = val.cuda(device=device_id)
      except:
        pass


def may_transfer_optims(optims, device_id=-1):
  """Transfer optimizers to cpu or specified gpu, which means transferring 
  tensors of the optimizer to specified device. The modification is in place 
  for the optimizers.
  Args:
    optims: A list, which members are either torch.nn.optimizer or None.
    device_id: gpu id, or -1 which means transferring to cpu
  """
  for optim in optims:
    if isinstance(optim, torch.optim.Optimizer):
      transfer_optim_state(optim.state, device_id=device_id)


def may_transfer_modules_optims(modules_and_or_optims, device_id=-1):
  """Transfer optimizers/modules to cpu or specified gpu.
  Args:
    modules_and_or_optims: A list, which members are either torch.nn.optimizer 
      or torch.nn.Module or None.
    device_id: gpu id, or -1 which means transferring to cpu
  """
  for item in modules_and_or_optims:
    if isinstance(item, torch.optim.Optimizer):
      transfer_optim_state(item.state, device_id=device_id)
    elif isinstance(item, torch.nn.Module):
      if device_id == -1:
        item.cpu()
      else:
        item.cuda(device_id=device_id)
    elif item is not None:
      print('[Warning] Invalid type {}'.format(item.__class__.__name__))


class TransferVarTensor(object):
  """Return a copy of the input Variable or Tensor on specified device."""

  def __init__(self, device_id=-1):
    self.device_id = device_id

  def __call__(self, var_or_tensor):
    return var_or_tensor.cpu() if self.device_id == -1 \
      else var_or_tensor.cuda(self.device_id)


class TransferModulesOptims(object):
  """Transfer optimizers/modules to cpu or specified gpu."""

  def __init__(self, device_id=-1):
    self.device_id = device_id

  def __call__(self, modules_and_or_optims):
    may_transfer_modules_optims(modules_and_or_optims, self.device_id)


def set_devices(sys_device_ids):
  """
  It sets some GPUs to be visible and returns some wrappers to transferring 
  Variables/Tensors and Modules/Optimizers.
  Args:
    sys_device_ids: which GPUs to use
      e.g.  sys_device_ids = (), only use cpu
            sys_device_ids = (3,), use the 4th gpu
            sys_device_ids = (0, 1, 2, 3,), use first 4 gpus
            sys_device_ids = (0, 2, 4,), use the 1st, 3rd and 5th gpus
  Returns:
    
  """
  # Set the CUDA_VISIBLE_DEVICES environment variable
  import os
  visible_devices = ''
  for i in sys_device_ids:
    visible_devices += '{}, '.format(i)
  os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
  # Return wrappers
  device_id = 0 if len(sys_device_ids) > 0 else -1
  TVT = TransferVarTensor(device_id)
  TMO = TransferModulesOptims(device_id)
  return TVT, TMO


def load_ckpt(modules_optims, ckpt_file, load_to_cpu=True, verbose=True):
  """Load state_dict's of modules/optimizers from file.
  Args:
    modules_optims: A list, which members are either torch.nn.optimizer 
      or torch.nn.Module.
    ckpt_file: The file path.
    load_to_cpu: Boolean. Whether to transform tensors in modules/optimizers 
      to cpu type.
  """
  map_location = (lambda storage, loc: storage) if load_to_cpu else None
  ckpt = torch.load(ckpt_file, map_location=map_location)
  for m, sd in zip(modules_optims, ckpt['state_dicts']):
    m.load_state_dict(sd)
  if verbose:
    print('Resume from ckpt {}, \nepoch {}, \nscores {}'.format(
      ckpt_file, ckpt['ep'], ckpt['scores']))
  return ckpt['ep'], ckpt['scores']


def save_ckpt(modules_optims, ep, scores, ckpt_file):
  """Save state_dict's of modules/optimizers to file. 
  Args:
    modules_optims: A list, which members are either torch.nn.optimizer 
      or torch.nn.Module.
    ep: the current epoch number
    scores: the performance of current model
    ckpt_file: The file path.
  Note:
    torch.save() reserves device type and id of tensors to save, so when 
    loading ckpt, you have to inform torch.load() to load these tensors to 
    cpu or your desired gpu, if you change devices.
  """
  state_dicts = [m.state_dict() for m in modules_optims]
  ckpt = dict(state_dicts=state_dicts,
              ep=ep,
              scores=scores)
  may_make_dir(osp.dirname(osp.abspath(ckpt_file)))
  torch.save(ckpt, ckpt_file)


def load_module_state_dict(model, state_dict):
  """Copies parameters and buffers from `state_dict` into `model` and its 
  descendants. The keys of `state_dict` NEED NOT exactly match the keys 
  returned by model's `state_dict()` function. For dict key mismatch, just
  skip it; for copying error, just output warnings and proceed.

  Arguments:
    model: A torch.nn.Module object. 
    state_dict (dict): A dict containing parameters and persistent buffers.
  Note:
    This is copied and modified from torch.nn.modules.module.load_state_dict().
    Just to allow name mismatch between `model.state_dict()` and `state_dict`.
  """
  import warnings
  from torch.nn import Parameter

  own_state = model.state_dict()
  for name, param in state_dict.items():
    if name not in own_state:
      warnings.warn('Skipping unexpected key "{}" in state_dict'.format(name))
      continue
    if isinstance(param, Parameter):
      # backwards compatibility for serialized parameters
      param = param.data
    try:
      own_state[name].copy_(param)
    except Exception, msg:
      warnings.warn("Error occurs when copying from state_dict['{}']: {}"
                    .format(name, str(msg)))

  missing = set(own_state.keys()) - set(state_dict.keys())
  if len(missing) > 0:
    warnings.warn(
      "Keys not found in state_dict and thus not overwritten: '{}'"
        .format(missing))


def may_set_mode(maybe_modules, mode):
  """maybe_modules: an object or a list of objects."""
  assert mode in ['train', 'eval']
  if not is_iterable(maybe_modules):
    maybe_modules = [maybe_modules]
  for m in maybe_modules:
    if isinstance(m, torch.nn.Module):
      if mode == 'train':
        m.train()
      else:
        m.eval()


def may_make_dir(path):
  """
  Args:
    path: a dir, or result of `osp.dirname(osp.abspath(file_path))`
  Note:
    `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
  """
  # This clause has mistakes:
  # if path is None or '':

  if path in [None, '']:
    return
  if not osp.exists(path):
    os.makedirs(path)


def is_iterable(obj):
  return hasattr(obj, '__len__')


def adjust_lr_staircase(param_groups=None, base_lrs=None,
                        decay_epochs=None, epoch=None, ratios=0.1,
                        verbose=False):
  """Decay the learning rates in a staircase manner.
  Args:
    param_groups: typically returned by `some_optimizer.param_groups`
    base_lrs: a scalar or a list    
    decay_epochs: a scalar or a list
    epoch: the current epoch number
    ratios: a scalar or a list
  Returns:
    lrs: the learning rates after adjusting
  """
  if not is_iterable(base_lrs):
    base_lrs = [base_lrs for _ in param_groups]
  if not is_iterable(decay_epochs):
    decay_epochs = [decay_epochs for _ in param_groups]
  if not is_iterable(ratios):
    ratios = [ratios for _ in param_groups]
  lrs = []
  log = 'lr adjusted to '
  for param_group, base_lr, decay_epoch, ratio in \
      zip(param_groups, base_lrs, decay_epochs, ratios):
    lr = base_lr * (ratio ** (epoch // decay_epoch))
    param_group['lr'] = lr
    lrs.append(lr)
    log += '{:.10f}'.format(param_group['lr']).rstrip('0').rstrip('.') + ', '
  if verbose:
    print(log)
  return lrs


def adjust_lr_poly(param_groups=None, base_lrs=None, total_epochs=None,
                   epoch=None, pows=0.5, verbose=False):
  """Decay the learning rates using a polynomial curve.
  Args:
    param_groups: typically returned by `some_optimizer.param_groups`
    base_lrs: a scalar or a list
    total_epochs: a scalar
    epoch: the current epoch number
    pows: a scalar or a list
  Returns:
    lrs: the learning rates after adjusting
  """
  if not is_iterable(base_lrs):
    base_lrs = [base_lrs for _ in param_groups]
  if not is_iterable(pows):
    pows = [pows for _ in param_groups]
  lrs = []
  log = 'lr adjusted to '
  for param_group, base_lr, pow in zip(param_groups, base_lrs, pows):
    lr = base_lr * np.power(float(total_epochs - epoch) / total_epochs, pow)
    param_group['lr'] = lr
    lrs.append(lr)
    log += '{:.10f}'.format(param_group['lr']).rstrip('0').rstrip('.') + ', '
  if verbose:
    print(log)
  return lrs


def adjust_lr(type='staircase', *args, **kwargs):
  assert type in ['staircase', 'poly']
  if type == 'staircase':
    adjust_lr_staircase(*args, **kwargs)
  elif type == 'poly':
    adjust_lr_poly(*args, **kwargs)
  else:
    raise NotImplementedError


def make_sure_str_list(may_be_list):
  if isinstance(may_be_list, str):
    may_be_list = [may_be_list]
  return may_be_list


def repeat_select_with_replacement(samples, num_select):
  """Repeat {select one sample with replacement} for n times.
  Args:
    samples: a numpy array with shape [num_samples]
    num_select: an int, number of selections
  Returns:
    all_select: a numpy array with shape [num_comb, num_select], where 
      num_comb is the number of all possible combinations
  """
  num_samples = len(samples)
  num_comb = num_samples ** num_select
  all_select = np.zeros([num_comb, num_select], dtype=samples.dtype)
  for i in range(num_select):
    num_repeat = num_samples ** (num_select - i - 1)
    # num_tile = num_comb / (num_repeat * num_samples)
    num_tile = num_samples ** i
    all_select[:, i] = np.tile(np.repeat(samples, num_repeat), num_tile)
  return all_select


# It seems Numpy already supports this functionality?
def index_select(mat, inds):
  """
  Args:
    mat: numpy array with shape [m, n]
    inds: numpy array with shape [m, n], e.g. the result from `np.argsort(mat)`
  Returns:
    ret: numpy array with shape [m, n]
  """
  ret = np.empty_like(mat)
  for i in range(ret.shape[0]):
    ret[i] = mat[i, inds[i]]
  return ret


class AverageMeter(object):
  """Modified from Tong Xiao's open-reid. 
  Computes and stores the average and current value"""

  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = float(self.sum) / (self.count + 1e-20)


class RunningAverageMeter(object):
  """Computes and stores the running average and current value"""

  def __init__(self, hist=0.99):
    self.val = None
    self.avg = None
    self.hist = hist

  def reset(self):
    self.val = None
    self.avg = None

  def update(self, val):
    if self.avg is None:
      self.avg = val
    else:
      self.avg = self.avg * self.hist + val * (1 - self.hist)
    self.val = val


class RecentAverageMeter(object):
  """Stores and computes the average of recent values."""

  def __init__(self, hist_size=100):
    self.hist_size = hist_size
    self.fifo = []
    self.val = 0

  def reset(self):
    self.fifo = []
    self.val = 0

  def update(self, val):
    self.val = val
    self.fifo.append(val)
    if len(self.fifo) > self.hist_size:
      del self.fifo[0]

  @property
  def avg(self):
    assert len(self.fifo) > 0
    return float(sum(self.fifo)) / len(self.fifo)


def get_model_wrapper(model, multi_gpu):
  from torch.nn.parallel import DataParallel
  if multi_gpu:
    return DataParallel(model)
  else:
    return model


class ReDirectSTD(object):
  """Modified from Tong Xiao's `Logger` in open-reid.
  This class overwrites sys.stdout or sys.stderr, so that console logs can
  also be written to file.
  Args:
    fpath: file path
    console: one of ['stdout', 'stderr']
    immediately_visible: If `False`, the file is opened only once and closed
      after exiting. In this case, the message written to file may not be
      immediately visible (Because the file handle is occupied by the
      program?). If `True`, each writing operation of the console will
      open, write to, and close the file. If your program has tons of writing
      operations, the cost of opening and closing file may be obvious. (?)
  Usage example:
    `ReDirectSTD('stdout.txt', 'stdout', False)`
    `ReDirectSTD('stderr.txt', 'stderr', False)`
  NOTE: File will be deleted if already existing. Log dir and file is created
    lazily -- if no message is written, the dir and file will not be created.
  """

  def __init__(self, fpath=None, console='stdout', immediately_visible=False):
    import sys
    import os
    import os.path as osp

    assert console in ['stdout', 'stderr']
    self.console = sys.stdout if console == 'stdout' else sys.stderr
    self.file = fpath
    self.f = None
    self.immediately_visible = immediately_visible
    if fpath is not None:
      # Remove existing log file.
      if osp.exists(fpath):
        os.remove(fpath)

    # Overwrite
    if console == 'stdout':
      sys.stdout = self
    else:
      sys.stderr = self

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
      may_make_dir(os.path.dirname(osp.abspath(self.file)))
      if self.immediately_visible:
        with open(self.file, 'a') as f:
          f.write(msg)
      else:
        if self.f is None:
          self.f = open(self.file, 'w')
        self.f.write(msg)

  def flush(self):
    self.console.flush()
    if self.f is not None:
      self.f.flush()
      import os
      os.fsync(self.f.fileno())

  def close(self):
    self.console.close()
    if self.f is not None:
      self.f.close()


class Logger(object):
  """Copied from Tong Xiao's open-reid. 
  This class overwrites sys.stdout or sys.stderr, so that console logs can 
  also be written to file.
  
  Usage example:
    import sys
    sys.stdout = Logger('stdout.txt', sys.stdout)
    sys.stderr = Logger('stderr.txt', sys.stderr)
  """

  def __init__(self, fpath=None, console=sys.stdout):
    assert console in [sys.stdout, sys.stderr]
    self.console = console
    self.file = None
    if fpath is not None:
      may_make_dir(os.path.dirname(osp.abspath(fpath)))
      self.file = open(fpath, 'w')

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
      self.file.write(msg)

  def flush(self):
    self.console.flush()
    if self.file is not None:
      self.file.flush()
      os.fsync(self.file.fileno())

  def close(self):
    self.console.close()
    if self.file is not None:
      self.file.close()


def set_seed(seed):
  import random
  random.seed(seed)
  print('setting random-seed to {}'.format(seed))

  import numpy as np
  np.random.seed(seed)
  print('setting np-random-seed to {}'.format(seed))

  import torch
  torch.backends.cudnn.enabled = False
  print('cudnn.enabled set to {}'.format(torch.backends.cudnn.enabled))
  # set seed for CPU
  torch.manual_seed(seed)
  print('setting torch-seed to {}'.format(seed))
  try:
    # set seed for all visible GPUs
    torch.cuda.manual_seed_all(seed)
    print('setting torch-cuda-seed to {}'.format(seed))
  except:
    pass


def softmax(x, T=1.):
  """T is the temperature. Higher T produces a softer probability distribution 
  over classes."""
  return np.exp(x / T) / np.sum(np.exp(x / T))


class ParseSeq(object):
  """Create a function to pass to argparse, so that it can parse input in 
  this way:
    '1' => (1,)
    '1,' => (1,)
    '1, 2' => (1, 2)
  """

  def __init__(self, split=',', func=int, ret_type=tuple):
    self.split = split
    self.func = func
    self.ret_type = ret_type

  def __call__(self, s):
    return self.ret_type(self.func(n) for n in s.split(self.split))


def thread_safe_append(file, msg):
  """Append a message to file, locking the file during writing. 
  NOTE: thread safe can only be achieved when all threads/processes obey this 
  acquiring lock rule (use fcntl.flock to acquire lock before writing)."""
  import fcntl
  may_make_dir(osp.dirname(osp.abspath(file)))
  with open(file, 'a') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(msg)
    fcntl.flock(f, fcntl.LOCK_UN)


def display_image_in_actual_size(im_path):
  import matplotlib.pyplot as plt

  dpi = 80
  im_data = plt.imread(im_path)
  height, width, depth = im_data.shape

  # What size does the figure need to be in inches to fit the image?
  figsize = width / float(dpi), height / float(dpi)

  # Create a figure of the right size with one axes that takes up the full figure
  fig = plt.figure(figsize=figsize)
  ax = fig.add_axes([0, 0, 1, 1])

  # Hide spines, ticks, etc.
  ax.axis('off')

  # Display the image.
  ax.imshow(im_data, cmap='gray')

  plt.show()


def read_lines(file):
  with open(file, 'r') as f:
    lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


def write_lines(file, lines):
  may_make_dir(osp.dirname(osp.abspath(file)))
  with open(file, 'w') as f:
    for l in lines:
      f.write(l + '\n')


def print_array(array, fmt='{:.2f}', end=' '):
  """Print a 1-D tuple, list, or numpy array containing digits."""
  s = ''
  for x in array:
    s += fmt.format(float(x)) + end
  s += '\n'
  print(s)
  return s


# Great idea from https://github.com/amdegroot/ssd.pytorch
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


def tight_float_str(x, fmt='{:.4f}'):
  return fmt.format(x).rstrip('0').rstrip('.')


def find_index(seq, item):
  for i, x in enumerate(seq):
    if item == x:
      return i
  return -1