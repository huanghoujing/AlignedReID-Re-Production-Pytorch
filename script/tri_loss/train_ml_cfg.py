from aligned_reid.utils.utils import load_pickle
from aligned_reid.utils.utils import time_str
from aligned_reid.utils.utils import str2bool
from aligned_reid.utils.utils import tight_float_str as tfs

import numpy as np
import argparse
import os.path as osp


class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=((0,),))
    parser.add_argument('--num_models', type=int, default=1)
    parser.add_argument('-r', '--run', type=int, default=1)
    parser.add_argument('--set_seed', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke'])
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])

    # Only for training set.
    parser.add_argument('--ids_per_batch', type=int, default=32)
    parser.add_argument('--ims_per_id', type=int, default=4)

    parser.add_argument('--log_to_file', type=str2bool, default=True)
    parser.add_argument('--normalize_feature', type=str2bool, default=True)
    parser.add_argument('--local_dist_own_hard_sample',
                        type=str2bool, default=False)
    parser.add_argument('-gm', '--global_margin', type=float, default=0.3)
    parser.add_argument('-lm', '--local_margin', type=float, default=0.3)
    parser.add_argument('-glw', '--g_loss_weight', type=float, default=1.)
    parser.add_argument('-llw', '--l_loss_weight', type=float, default=0.)
    parser.add_argument('-idlw', '--id_loss_weight', type=float, default=0.)
    parser.add_argument('-pmlw', '--pm_loss_weight', type=float, default=1.)
    parser.add_argument('-gdmlw', '--gdm_loss_weight', type=float, default=1.)
    parser.add_argument('-ldmlw', '--ldm_loss_weight', type=float, default=0.)

    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--exp_dir', type=str, default='')

    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--lr_decay_type', type=str, default='exp',
                        choices=['exp', 'staircase'])
    parser.add_argument('--exp_decay_at_epoch', type=int, default=76)
    parser.add_argument('--staircase_decay_at_epochs',
                        type=str, default='(101, 201,)')
    parser.add_argument('--staircase_decay_multiply_factor',
                        type=float, default=0.1)
    parser.add_argument('--total_epochs', type=int, default=150)

    args = parser.parse_known_args()[0]

    # gpu ids
    self.sys_device_ids = args.sys_device_ids

    if args.set_seed:
      self.seed = 1
    else:
      self.seed = None

    # The experiments can be run for several times and performances be averaged.
    # `run` starts from `1`, not `0`.
    self.run = args.run

    ###########
    # Dataset #
    ###########

    # If you want to exactly reproduce the result in training, you have to set
    # num of threads to 1.
    if self.seed is not None:
      self.prefetch_threads = 1
    else:
      self.prefetch_threads = 2

    self.dataset = args.dataset

    if self.dataset == 'market1501':
      self.im_dir = osp.expanduser('~/Dataset/market1501/images')
      self.partition_file = osp.expanduser(
        '~/Dataset/market1501/partitions.pkl')
    elif self.dataset == 'cuhk03':
      self.im_type = ['detected', 'labeled'][0]
      self.im_dir = osp.expanduser(
        osp.join('~/Dataset/cuhk03', self.im_type, 'images'))
      self.partition_file = osp.expanduser(
        osp.join('~/Dataset/cuhk03', self.im_type, 'partitions.pkl'))
    elif self.dataset == 'duke':
      self.im_dir = osp.expanduser('~/Dataset/duke/images')
      self.partition_file = osp.expanduser('~/Dataset/duke/partitions.pkl')

    self.trainset_part = args.trainset_part
    # num of classes in reid net.
    self.num_classes = \
      len(load_pickle(self.partition_file)[self.trainset_part + '_ids2labels'])

    # Image Processing

    # (width, height)
    self.im_resize_size = (128, 256)
    self.im_crop_size = (128, 256)
    # Whether to scale by 1/255
    self.scale_im = True
    self.im_mean = [0.486, 0.459, 0.408]
    # Whether to divide by std, set to `None` to disable.
    # Dividing is applied only when subtracting mean is applied.
    self.im_std = [0.229, 0.224, 0.225]

    self.ids_per_batch = args.ids_per_batch
    self.ims_per_id = args.ims_per_id
    self.train_final_batch = True
    self.train_mirror_type = ['random', 'always', None][0]
    self.train_shuffle = True

    self.test_batch_size = 32
    self.test_final_batch = True
    self.test_mirror_type = ['random', 'always', None][2]
    self.test_shuffle = False

    dataset_kwargs = dict(
      name=self.dataset,
      partition_file=self.partition_file,
      im_dir=self.im_dir,
      resize_size=self.im_resize_size,
      crop_size=self.im_crop_size,
      scale=self.scale_im,
      im_mean=self.im_mean,
      im_std=self.im_std,
      batch_dims='NCHW',
      num_prefetch_threads=self.prefetch_threads)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.train_set_kwargs = dict(
      part=self.trainset_part,
      ids_per_batch=self.ids_per_batch,
      ims_per_id=self.ims_per_id,
      final_batch=self.train_final_batch,
      shuffle=self.train_shuffle,
      mirror_type=self.train_mirror_type,
      prng=prng)
    self.train_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.test_set_kwargs = dict(
      part='test',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.test_set_kwargs.update(dataset_kwargs)

    ###############
    # ReID Model  #
    ###############

    self.local_dist_own_hard_sample = args.local_dist_own_hard_sample

    self.normalize_feature = args.normalize_feature

    self.local_conv_out_channels = 128
    self.global_margin = args.global_margin
    self.local_margin = args.local_margin

    # Identification Loss weight
    self.id_loss_weight = args.id_loss_weight

    # global loss weight
    self.g_loss_weight = args.g_loss_weight
    # local loss weight
    self.l_loss_weight = args.l_loss_weight

    ###############
    # Mutual Loss #
    ###############

    # probability mutual loss weight
    self.pm_loss_weight = args.pm_loss_weight
    # global distance mutual loss weight
    self.gdm_loss_weight = args.gdm_loss_weight
    # local distance mutual loss weight
    self.ldm_loss_weight = args.ldm_loss_weight

    self.num_models = args.num_models
    # See method `set_devices_for_ml` in `aligned_reid/utils/utils.py` for
    # details.
    assert len(self.sys_device_ids) == self.num_models, \
      'You should specify device for each model.'

    # Currently one model occupying multiple GPUs is not allowed.
    if self.num_models > 1:
      for ids in self.sys_device_ids:
        assert len(ids) == 1, "When num_models > 1, one model occupying " \
                              "multiple GPUs is not allowed."

    #############
    # Training  #
    #############

    self.weight_decay = 0.0005

    # Initial learning rate
    self.base_lr = args.base_lr
    self.lr_decay_type = args.lr_decay_type
    self.exp_decay_at_epoch = args.exp_decay_at_epoch
    self.staircase_decay_at_epochs = eval(args.staircase_decay_at_epochs)
    self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
    # Number of epochs to train
    self.total_epochs = args.total_epochs

    # How often (in batches) to log. If only need to log the average
    # information for each epoch, set this to a large value, e.g. 1e10.
    self.log_steps = 1e10

    # Only test and without training.
    self.only_test = args.only_test

    self.resume = args.resume

    #######
    # Log #
    #######

    self.log_to_file = args.log_to_file

    # The root dir of logs.
    if args.exp_dir == '':
      self.exp_dir = osp.join(
        'exp/tri_loss',
        '{}'.format(self.dataset),
        'train',
        #
        ('nf_' if self.normalize_feature else 'not_nf_') +
        ('ohs_' if self.local_dist_own_hard_sample else 'not_ohs_') +
        'gm_{}_'.format(tfs(self.global_margin)) +
        'lm_{}_'.format(tfs(self.local_margin)) +
        'glw_{}_'.format(tfs(self.g_loss_weight)) +
        'llw_{}_'.format(tfs(self.l_loss_weight)) +
        'idlw_{}_'.format(tfs(self.id_loss_weight)) +
        'pmlw_{}_'.format(tfs(self.pm_loss_weight)) +
        'gdmlw_{}_'.format(tfs(self.gdm_loss_weight)) +
        'ldmlw_{}_'.format(tfs(self.ldm_loss_weight)) +
        'lr_{}_'.format(tfs(self.base_lr)) +
        '{}_'.format(self.lr_decay_type) +
        ('decay_at_{}_'.format(self.exp_decay_at_epoch)
         if self.lr_decay_type == 'exp'
         else 'decay_at_{}_factor_{}_'.format(
          args.staircase_decay_at_epochs,
          tfs(self.staircase_decay_multiply_factor))) +
        'total_{}'.format(self.total_epochs),
        #
        'run{}'.format(self.run),
      )
    else:
      self.exp_dir = args.exp_dir

  @property
  def log_file(self):
    return osp.join(self.exp_dir, 'log_stdout_{}.txt'.format(time_str()))

  @property
  def log_err_file(self):
    return osp.join(self.exp_dir, 'log_stderr_{}.txt'.format(time_str()))

  @property
  def ckpt_file(self):
    return osp.join(self.exp_dir, 'ckpt.pth')
