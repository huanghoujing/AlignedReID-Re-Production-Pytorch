import os.path as osp
from aligned_reid.utils.utils import load_pickle
from aligned_reid.utils.utils import time_str
import numpy as np
import argparse


class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids',
                        type=str, default='(0,)')
    parser.add_argument('-r', '--run', type=int, default=1)
    parser.add_argument('--set_seed', action='store_true')
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke'])
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])
    parser.add_argument('--log_to_file', action='store_true')
    parser.add_argument('-glw', '--g_loss_weight', type=float, default=1.)
    parser.add_argument('-llw', '--l_loss_weight', type=float, default=0.)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--exp_dir', type=str, default='')

    args = parser.parse_known_args()[0]

    # gpu ids
    self.sys_device_ids = eval(args.sys_device_ids)

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

    self.ids_per_batch = 32
    self.ims_per_id = 4
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

    # prng = np.random
    # if self.seed is not None:
    #   prng = np.random.RandomState(self.seed)
    # self.val_set_kwargs = dict(
    #   part='val',
    #   batch_size=self.test_batch_size,
    #   final_batch=self.test_final_batch,
    #   shuffle=self.test_shuffle,
    #   mirror_type=self.test_mirror_type,
    #   prng=prng)
    # self.val_set_kwargs.update(dataset_kwargs)

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

    self.local_conv_out_channels = 128
    self.global_margin = 0.3
    self.local_margin = 0.3

    # global loss weight
    self.g_loss_weight = args.g_loss_weight
    # local loss weight
    self.l_loss_weight = args.l_loss_weight

    #######
    # Log #
    #######

    self.log_to_file = args.log_to_file

    # The root dir of logs.
    if args.exp_dir == '':
      self.exp_dir = ('exp/tri_loss/{}/train/g{:.4f}_l{:.4f}/run{}'
        .format(self.dataset, self.g_loss_weight, self.l_loss_weight, self.run))
    else:
      self.exp_dir = args.exp_dir

    #############
    # Training  #
    #############

    self.weight_decay = 0.0005
    # Initial learning rate
    self.lr = 2e-4
    self.start_decay_epoch = 100
    # Number of epochs to train
    self.num_epochs = 150
    # How often (in batches) to log. If only need to log the average
    # information for each epoch, set this to a large value, e.g. 1e10.
    self.log_steps = 1e10

    # Only test and without training.
    self.only_test = args.only_test
    # Test after training.
    self.test = True

    self.save_ckpt = True

    self.resume = False

  @property
  def log_file(self):
    return osp.join(self.exp_dir, 'log_stdout_{}.txt'.format(time_str()))

  @property
  def log_err_file(self):
    return osp.join(self.exp_dir, 'log_stderr_{}.txt'.format(time_str()))

  @property
  def test_scores_file(self):
    """Test scores of different runs are appended to this global file."""
    return osp.join(osp.dirname(self.exp_dir.rstrip('/')), 'test_scores.txt')

  @property
  def ckpt_file(self):
    return osp.join(self.exp_dir, 'ckpt.pth')

  @property
  def best_ckpt_file(self):
    return osp.join(self.exp_dir, 'best_ckpt.pth')