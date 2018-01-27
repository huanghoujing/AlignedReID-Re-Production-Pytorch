from __future__ import print_function
import os
import os.path as osp
import cPickle as pickle
from scipy import io
import datetime
import time
from contextlib import contextmanager

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
        item.cuda(device=device_id)
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
    sys_device_ids: a tuple; which GPUs to use
      e.g.  sys_device_ids = (), only use cpu
            sys_device_ids = (3,), use the 4th gpu
            sys_device_ids = (0, 1, 2, 3,), use first 4 gpus
            sys_device_ids = (0, 2, 4,), use the 1st, 3rd and 5th gpus
  Returns:
    TVT: a `TransferVarTensor` callable
    TMO: a `TransferModulesOptims` callable
  """
  # Set the CUDA_VISIBLE_DEVICES environment variable
  import os
  visible_devices = ''
  for i in sys_device_ids:
    visible_devices += '{}, '.format(i)
  os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
  # Return wrappers.
  # Models and user defined Variables/Tensors would be transferred to the
  # first device.
  device_id = 0 if len(sys_device_ids) > 0 else -1
  TVT = TransferVarTensor(device_id)
  TMO = TransferModulesOptims(device_id)
  return TVT, TMO


def set_devices_for_ml(sys_device_ids):
  """This version is for mutual learning.
  
  It sets some GPUs to be visible and returns some wrappers to transferring 
  Variables/Tensors and Modules/Optimizers.
  
  Args:
    sys_device_ids: a tuple of tuples; which devices to use for each model, 
      len(sys_device_ids) should be equal to number of models. Examples:
        
      sys_device_ids = ((-1,), (-1,))
        the two models both on CPU
      sys_device_ids = ((-1,), (2,))
        the 1st model on CPU, the 2nd model on GPU 2
      sys_device_ids = ((3,),)
        the only one model on the 4th gpu 
      sys_device_ids = ((0, 1), (2, 3))
        the 1st model on GPU 0 and 1, the 2nd model on GPU 2 and 3
      sys_device_ids = ((0,), (0,))
        the two models both on GPU 0
      sys_device_ids = ((0,), (0,), (1,), (1,))
        the 1st and 2nd model on GPU 0, the 3rd and 4th model on GPU 1
  
  Returns:
    TVTs: a list of `TransferVarTensor` callables, one for one model.
    TMOs: a list of `TransferModulesOptims` callables, one for one model.
    relative_device_ids: a list of lists; `sys_device_ids` transformed to 
      relative ids; to be used in `DataParallel`
  """
  import os

  all_ids = []
  for ids in sys_device_ids:
    all_ids += ids
  unique_sys_device_ids = list(set(all_ids))
  unique_sys_device_ids.sort()
  if -1 in unique_sys_device_ids:
    unique_sys_device_ids.remove(-1)

  # Set the CUDA_VISIBLE_DEVICES environment variable

  visible_devices = ''
  for i in unique_sys_device_ids:
    visible_devices += '{}, '.format(i)
  os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices

  # Return wrappers

  relative_device_ids = []
  TVTs, TMOs = [], []
  for ids in sys_device_ids:
    relative_ids = []
    for id in ids:
      if id != -1:
        id = find_index(unique_sys_device_ids, id)
      relative_ids.append(id)
    relative_device_ids.append(relative_ids)

    # Models and user defined Variables/Tensors would be transferred to the
    # first device.
    TVTs.append(TransferVarTensor(relative_ids[0]))
    TMOs.append(TransferModulesOptims(relative_ids[0]))
  return TVTs, TMOs, relative_device_ids


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


def load_state_dict(model, src_state_dict):
  """Copy parameters and buffers from `src_state_dict` into `model` and its 
  descendants. The `src_state_dict.keys()` NEED NOT exactly match 
  `model.state_dict().keys()`. For dict key mismatch, just
  skip it; for copying error, just output warnings and proceed.

  Arguments:
    model: A torch.nn.Module object. 
    src_state_dict (dict): A dict containing parameters and persistent buffers.
  Note:
    This is modified from torch.nn.modules.module.load_state_dict(), to make
    the warnings and errors more detailed.
  """
  from torch.nn import Parameter

  dest_state_dict = model.state_dict()
  for name, param in src_state_dict.items():
    if name not in dest_state_dict:
      continue
    if isinstance(param, Parameter):
      # backwards compatibility for serialized parameters
      param = param.data
    try:
      dest_state_dict[name].copy_(param)
    except Exception, msg:
      print("Warning: Error occurs when copying '{}': {}"
            .format(name, str(msg)))

  src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
  if len(src_missing) > 0:
    print("Keys not found in source state_dict: ")
    for n in src_missing:
      print('\t', n)

  dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
  if len(dest_missing) > 0:
    print("Keys not found in destination state_dict: ")
    for n in dest_missing:
      print('\t', n)


def is_iterable(obj):
  return hasattr(obj, '__len__')


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


def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
  """Decay exponentially in the later phase of training. All parameters in the 
  optimizer share the same learning rate.
  
  Args:
    optimizer: a pytorch `Optimizer` object
    base_lr: starting learning rate
    ep: current epoch, ep >= 1
    total_ep: total number of epochs to train
    start_decay_at_ep: start decaying at the BEGINNING of this epoch
  
  Example:
    base_lr = 2e-4
    total_ep = 300
    start_decay_at_ep = 201
    It means the learning rate starts at 2e-4 and begins decaying after 200 
    epochs. And training stops after 300 epochs.
  
  NOTE: 
    It is meant to be called at the BEGINNING of an epoch.
  """
  assert ep >= 1, "Current epoch number should be >= 1"

  if ep < start_decay_at_ep:
    return

  for g in optimizer.param_groups:
    g['lr'] = (base_lr * (0.001 ** (float(ep + 1 - start_decay_at_ep)
                                    / (total_ep + 1 - start_decay_at_ep))))
  print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


def adjust_lr_staircase(optimizer, base_lr, ep, decay_at_epochs, factor):
  """Multiplied by a factor at the BEGINNING of specified epochs. All 
  parameters in the optimizer share the same learning rate.
  
  Args:
    optimizer: a pytorch `Optimizer` object
    base_lr: starting learning rate
    ep: current epoch, ep >= 1
    decay_at_epochs: a list or tuple; learning rate is multiplied by a factor 
      at the BEGINNING of these epochs
    factor: a number in range (0, 1)
  
  Example:
    base_lr = 1e-3
    decay_at_epochs = [51, 101]
    factor = 0.1
    It means the learning rate starts at 1e-3 and is multiplied by 0.1 at the 
    BEGINNING of the 51'st epoch, and then further multiplied by 0.1 at the 
    BEGINNING of the 101'st epoch, then stays unchanged till the end of 
    training.
  
  NOTE: 
    It is meant to be called at the BEGINNING of an epoch.
  """
  assert ep >= 1, "Current epoch number should be >= 1"

  if ep not in decay_at_epochs:
    return

  ind = find_index(decay_at_epochs, ep)
  for g in optimizer.param_groups:
    g['lr'] = base_lr * factor ** (ind + 1)
  print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


@contextmanager
def measure_time(enter_msg):
  st = time.time()
  print(enter_msg)
  yield
  print('Done, {:.2f}s'.format(time.time() - st))