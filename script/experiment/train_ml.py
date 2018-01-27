"""Train with optional Global Distance, Local Distance, Identification Loss, 
Mutual Loss."""
from __future__ import print_function

import sys
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

import time
import os.path as osp
from tensorboardX import SummaryWriter
import threading
import numpy as np
import argparse

from aligned_reid.dataset import create_dataset
from aligned_reid.model.Model import Model
from aligned_reid.model.TripletLoss import TripletLoss
from aligned_reid.model.loss import global_loss
from aligned_reid.model.loss import local_loss

from aligned_reid.utils.utils import time_str
from aligned_reid.utils.utils import str2bool
from aligned_reid.utils.utils import tight_float_str as tfs
from aligned_reid.utils.utils import may_set_mode
from aligned_reid.utils.utils import load_ckpt
from aligned_reid.utils.utils import save_ckpt
from aligned_reid.utils.utils import set_devices_for_ml
from aligned_reid.utils.utils import AverageMeter
from aligned_reid.utils.utils import to_scalar
from aligned_reid.utils.utils import ReDirectSTD
from aligned_reid.utils.utils import set_seed
from aligned_reid.utils.utils import adjust_lr_exp
from aligned_reid.utils.utils import adjust_lr_staircase


class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=((0,),))
    parser.add_argument('--num_models', type=int, default=1)
    parser.add_argument('-r', '--run', type=int, default=1)
    parser.add_argument('--set_seed', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke', 'combined'])
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])

    # Only for training set.
    parser.add_argument('--resize_h_w', type=eval, default=(256, 128))
    parser.add_argument('--crop_prob', type=float, default=0)
    parser.add_argument('--crop_ratio', type=float, default=1)
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
                        type=eval, default=(101, 201,))
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
    self.trainset_part = args.trainset_part

    # Image Processing

    # Just for training set
    self.crop_prob = args.crop_prob
    self.crop_ratio = args.crop_ratio
    self.resize_h_w = args.resize_h_w

    # Whether to scale by 1/255
    self.scale_im = True
    self.im_mean = [0.486, 0.459, 0.408]
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
      resize_h_w=self.resize_h_w,
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
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
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

    #############
    # Training  #
    #############

    self.weight_decay = 0.0005

    # Initial learning rate
    self.base_lr = args.base_lr
    self.lr_decay_type = args.lr_decay_type
    self.exp_decay_at_epoch = args.exp_decay_at_epoch
    self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
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

    # If True,
    # 1) stdout and stderr will be redirected to file,
    # 2) training loss etc will be written to tensorboard,
    # 3) checkpoint will be saved
    self.log_to_file = args.log_to_file

    # The root dir of logs.
    if args.exp_dir == '':
      self.exp_dir = osp.join(
        'exp/train_ml',
        '{}'.format(self.dataset),
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
          '_'.join([str(e) for e in args.staircase_decay_at_epochs]),
          tfs(self.staircase_decay_multiply_factor))) +
        'total_{}'.format(self.total_epochs),
        #
        'run{}'.format(self.run),
      )
    else:
      self.exp_dir = args.exp_dir

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str()))
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str()))

    # Saving model weights and optimizer states, for resuming.
    self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')


class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()
    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    global_feat, local_feat = self.model(ims)[:2]
    global_feat = global_feat.data.cpu().numpy()
    local_feat = local_feat.data.cpu().numpy()
    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return global_feat, local_feat


def main():
  cfg = Config()

  # Redirect logs to both console and file.
  if cfg.log_to_file:
    ReDirectSTD(cfg.stdout_file, 'stdout', False)
    ReDirectSTD(cfg.stderr_file, 'stderr', False)

  # Lazily create SummaryWriter
  writer = None

  TVTs, TMOs, relative_device_ids = set_devices_for_ml(cfg.sys_device_ids)

  if cfg.seed is not None:
    set_seed(cfg.seed)

  # Dump the configurations to log.
  import pprint
  print('-' * 60)
  print('cfg.__dict__')
  pprint.pprint(cfg.__dict__)
  print('-' * 60)

  ###########
  # Dataset #
  ###########

  train_set = create_dataset(**cfg.train_set_kwargs)

  test_sets = []
  test_set_names = []
  if cfg.dataset == 'combined':
    for name in ['market1501', 'cuhk03', 'duke']:
      cfg.test_set_kwargs['name'] = name
      test_sets.append(create_dataset(**cfg.test_set_kwargs))
      test_set_names.append(name)
  else:
    test_sets.append(create_dataset(**cfg.test_set_kwargs))
    test_set_names.append(cfg.dataset)

  ###########
  # Models  #
  ###########

  models = [Model(local_conv_out_channels=cfg.local_conv_out_channels,
                  num_classes=len(train_set.ids2labels))
            for _ in range(cfg.num_models)]
  # Model wrappers
  model_ws = [DataParallel(models[i], device_ids=relative_device_ids[i])
              for i in range(cfg.num_models)]

  #############################
  # Criteria and Optimizers   #
  #############################

  id_criterion = nn.CrossEntropyLoss()
  g_tri_loss = TripletLoss(margin=cfg.global_margin)
  l_tri_loss = TripletLoss(margin=cfg.local_margin)

  optimizers = [optim.Adam(m.parameters(),
                           lr=cfg.base_lr,
                           weight_decay=cfg.weight_decay)
                for m in models]

  # Bind them together just to save some codes in the following usage.
  modules_optims = models + optimizers

  ################################
  # May Resume Models and Optims #
  ################################

  if cfg.resume:
    resume_ep, scores = load_ckpt(modules_optims, cfg.ckpt_file)

  # May Transfer Models and Optims to Specified Device. Transferring optimizers
  # is to cope with the case when you load the checkpoint to a new device.
  for TMO, model, optimizer in zip(TMOs, models, optimizers):
    TMO([model, optimizer])

  ########
  # Test #
  ########

  # Test each model using different distance settings.
  def test(load_model_weight=False):
    if load_model_weight:
      load_ckpt(modules_optims, cfg.ckpt_file)

    use_local_distance = (cfg.l_loss_weight > 0) \
                         and cfg.local_dist_own_hard_sample

    for i, (model_w, TVT) in enumerate(zip(model_ws, TVTs)):
      for test_set, name in zip(test_sets, test_set_names):
        test_set.set_feat_func(ExtractFeature(model_w, TVT))
        print('\n=========> Test Model #{} on dataset: {} <=========\n'
              .format(i + 1, name))
        test_set.eval(
          normalize_feat=cfg.normalize_feature,
          use_local_distance=use_local_distance)

  if cfg.only_test:
    test(load_model_weight=True)
    return

  ############
  # Training #
  ############

  # Storing things that can be accessed cross threads.

  ims_list = [None for _ in range(cfg.num_models)]
  labels_list = [None for _ in range(cfg.num_models)]

  done_list1 = [False for _ in range(cfg.num_models)]
  done_list2 = [False for _ in range(cfg.num_models)]

  probs_list = [None for _ in range(cfg.num_models)]
  g_dist_mat_list = [None for _ in range(cfg.num_models)]
  l_dist_mat_list = [None for _ in range(cfg.num_models)]

  # Two phases for each model:
  # 1) forward and single-model loss;
  # 2) further add mutual loss and backward.
  # The 2nd phase is only ready to start when the 1st is finished for
  # all models.
  run_event1 = threading.Event()
  run_event2 = threading.Event()

  # This event is meant to be set to stop threads. However, as I found, with
  # `daemon` set to true when creating threads, manually stopping is
  # unnecessary. I guess some main-thread variables required by sub-threads
  # are destroyed when the main thread ends, thus the sub-threads throw errors
  # and exit too.
  # Real reason should be further explored.
  exit_event = threading.Event()

  # The function to be called by threads.
  def thread_target(i):
    while not exit_event.isSet():
      # If the run event is not set, the thread just waits.
      if not run_event1.wait(0.001): continue

      ######################################
      # Phase 1: Forward and Separate Loss #
      ######################################

      TVT = TVTs[i]
      model_w = model_ws[i]
      ims = ims_list[i]
      labels = labels_list[i]
      optimizer = optimizers[i]

      ims_var = Variable(TVT(torch.from_numpy(ims).float()))
      labels_t = TVT(torch.from_numpy(labels).long())
      labels_var = Variable(labels_t)

      global_feat, local_feat, logits = model_w(ims_var)
      probs = F.softmax(logits, dim=1)
      log_probs = F.log_softmax(logits, dim=1)

      g_loss, p_inds, n_inds, g_dist_ap, g_dist_an, g_dist_mat = global_loss(
        g_tri_loss, global_feat, labels_t,
        normalize_feature=cfg.normalize_feature)

      if cfg.l_loss_weight == 0:
        l_loss, l_dist_mat = 0, 0
      elif cfg.local_dist_own_hard_sample:
        # Let local distance find its own hard samples.
        l_loss, l_dist_ap, l_dist_an, l_dist_mat = local_loss(
          l_tri_loss, local_feat, None, None, labels_t,
          normalize_feature=cfg.normalize_feature)
      else:
        l_loss, l_dist_ap, l_dist_an = local_loss(
          l_tri_loss, local_feat, p_inds, n_inds, labels_t,
          normalize_feature=cfg.normalize_feature)
        l_dist_mat = 0

      id_loss = 0
      if cfg.id_loss_weight > 0:
        id_loss = id_criterion(logits, labels_var)

      probs_list[i] = probs
      g_dist_mat_list[i] = g_dist_mat
      l_dist_mat_list[i] = l_dist_mat

      done_list1[i] = True

      # Wait for event to be set, meanwhile checking if need to exit.
      while True:
        phase2_ready = run_event2.wait(0.001)
        if exit_event.isSet():
          return
        if phase2_ready:
          break

      #####################################
      # Phase 2: Mutual Loss and Backward #
      #####################################

      # Probability Mutual Loss (KL Loss)
      pm_loss = 0
      if (cfg.num_models > 1) and (cfg.pm_loss_weight > 0):
        for j in range(cfg.num_models):
          if j != i:
            pm_loss += F.kl_div(log_probs, TVT(probs_list[j]).detach(), False)
        pm_loss /= 1. * (cfg.num_models - 1) * len(ims)

      # Global Distance Mutual Loss (L2 Loss)
      gdm_loss = 0
      if (cfg.num_models > 1) and (cfg.gdm_loss_weight > 0):
        for j in range(cfg.num_models):
          if j != i:
            gdm_loss += torch.sum(torch.pow(
              g_dist_mat - TVT(g_dist_mat_list[j]).detach(), 2))
        gdm_loss /= 1. * (cfg.num_models - 1) * len(ims) * len(ims)

      # Local Distance Mutual Loss (L2 Loss)
      ldm_loss = 0
      if (cfg.num_models > 1) \
          and cfg.local_dist_own_hard_sample \
          and (cfg.ldm_loss_weight > 0):
        for j in range(cfg.num_models):
          if j != i:
            ldm_loss += torch.sum(torch.pow(
              l_dist_mat - TVT(l_dist_mat_list[j]).detach(), 2))
        ldm_loss /= 1. * (cfg.num_models - 1) * len(ims) * len(ims)

      loss = g_loss * cfg.g_loss_weight \
             + l_loss * cfg.l_loss_weight \
             + id_loss * cfg.id_loss_weight \
             + pm_loss * cfg.pm_loss_weight \
             + gdm_loss * cfg.gdm_loss_weight \
             + ldm_loss * cfg.ldm_loss_weight

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      ##################################
      # Step Log For One of the Models #
      ##################################

      # Just record for the first model
      if i == 0:

        # precision
        g_prec = (g_dist_an > g_dist_ap).data.float().mean()
        # the proportion of triplets that satisfy margin
        g_m = (g_dist_an > g_dist_ap + cfg.global_margin).data.float().mean()
        g_d_ap = g_dist_ap.data.mean()
        g_d_an = g_dist_an.data.mean()

        # These meters are outer-scope objects
        g_prec_meter.update(g_prec)
        g_m_meter.update(g_m)
        g_dist_ap_meter.update(g_d_ap)
        g_dist_an_meter.update(g_d_an)
        g_loss_meter.update(to_scalar(g_loss))

        if cfg.l_loss_weight > 0:
          # precision
          l_prec = (l_dist_an > l_dist_ap).data.float().mean()
          # the proportion of triplets that satisfy margin
          l_m = (l_dist_an > l_dist_ap + cfg.local_margin).data.float().mean()
          l_d_ap = l_dist_ap.data.mean()
          l_d_an = l_dist_an.data.mean()

          l_prec_meter.update(l_prec)
          l_m_meter.update(l_m)
          l_dist_ap_meter.update(l_d_ap)
          l_dist_an_meter.update(l_d_an)
          l_loss_meter.update(to_scalar(l_loss))

        if cfg.id_loss_weight > 0:
          id_loss_meter.update(to_scalar(id_loss))

        if (cfg.num_models > 1) and (cfg.pm_loss_weight > 0):
          pm_loss_meter.update(to_scalar(pm_loss))

        if (cfg.num_models > 1) and (cfg.gdm_loss_weight > 0):
          gdm_loss_meter.update(to_scalar(gdm_loss))

        if (cfg.num_models > 1) \
            and cfg.local_dist_own_hard_sample \
            and (cfg.ldm_loss_weight > 0):
          ldm_loss_meter.update(to_scalar(ldm_loss))

        loss_meter.update(to_scalar(loss))

      ###################
      # End Up One Step #
      ###################

      run_event1.clear()
      run_event2.clear()

      done_list2[i] = True

  threads = []
  for i in range(cfg.num_models):
    thread = threading.Thread(target=thread_target, args=(i,))
    # Set the thread in daemon mode, so that the main program ends normally.
    thread.daemon = True
    thread.start()
    threads.append(thread)

  start_ep = resume_ep if cfg.resume else 0
  for ep in range(start_ep, cfg.total_epochs):

    # Adjust Learning Rate
    for optimizer in optimizers:
      if cfg.lr_decay_type == 'exp':
        adjust_lr_exp(
          optimizer,
          cfg.base_lr,
          ep + 1,
          cfg.total_epochs,
          cfg.exp_decay_at_epoch)
      else:
        adjust_lr_staircase(
          optimizer,
          cfg.base_lr,
          ep + 1,
          cfg.staircase_decay_at_epochs,
          cfg.staircase_decay_multiply_factor)

    may_set_mode(modules_optims, 'train')

    epoch_done = False

    g_prec_meter = AverageMeter()
    g_m_meter = AverageMeter()
    g_dist_ap_meter = AverageMeter()
    g_dist_an_meter = AverageMeter()
    g_loss_meter = AverageMeter()

    l_prec_meter = AverageMeter()
    l_m_meter = AverageMeter()
    l_dist_ap_meter = AverageMeter()
    l_dist_an_meter = AverageMeter()
    l_loss_meter = AverageMeter()

    id_loss_meter = AverageMeter()

    # Global Distance Mutual Loss
    gdm_loss_meter = AverageMeter()
    # Local Distance Mutual Loss
    ldm_loss_meter = AverageMeter()
    # Probability Mutual Loss
    pm_loss_meter = AverageMeter()

    loss_meter = AverageMeter()

    ep_st = time.time()
    step = 0
    while not epoch_done:

      step += 1
      step_st = time.time()

      ims, im_names, labels, mirrored, epoch_done = train_set.next_batch()

      for i in range(cfg.num_models):
        ims_list[i] = ims
        labels_list[i] = labels
        done_list1[i] = False
        done_list2[i] = False

      run_event1.set()
      # Waiting for phase 1 done
      while not all(done_list1): continue

      run_event2.set()
      # Waiting for phase 2 done
      while not all(done_list2): continue

      ############
      # Step Log #
      ############

      if step % cfg.log_steps == 0:
        time_log = '\tStep {}/Ep {}, {:.2f}s'.format(
          step, ep + 1, time.time() - step_st, )

        if cfg.g_loss_weight > 0:
          g_log = (', gp {:.2%}, gm {:.2%}, '
                   'gd_ap {:.4f}, gd_an {:.4f}, '
                   'gL {:.4f}'.format(
            g_prec_meter.val, g_m_meter.val,
            g_dist_ap_meter.val, g_dist_an_meter.val,
            g_loss_meter.val, ))
        else:
          g_log = ''

        if cfg.l_loss_weight > 0:
          l_log = (', lp {:.2%}, lm {:.2%}, '
                   'ld_ap {:.4f}, ld_an {:.4f}, '
                   'lL {:.4f}'.format(
            l_prec_meter.val, l_m_meter.val,
            l_dist_ap_meter.val, l_dist_an_meter.val,
            l_loss_meter.val, ))
        else:
          l_log = ''

        if cfg.id_loss_weight > 0:
          id_log = (', idL {:.4f}'.format(id_loss_meter.val))
        else:
          id_log = ''

        if (cfg.num_models > 1) and (cfg.pm_loss_weight > 0):
          pm_log = (', pmL {:.4f}'.format(pm_loss_meter.val))
        else:
          pm_log = ''

        if (cfg.num_models > 1) and (cfg.gdm_loss_weight > 0):
          gdm_log = (', gdmL {:.4f}'.format(gdm_loss_meter.val))
        else:
          gdm_log = ''

        if (cfg.num_models > 1) \
            and cfg.local_dist_own_hard_sample \
            and (cfg.ldm_loss_weight > 0):
          ldm_log = (', ldmL {:.4f}'.format(ldm_loss_meter.val))
        else:
          ldm_log = ''

        total_loss_log = ', loss {:.4f}'.format(loss_meter.val)

        log = time_log + \
              g_log + l_log + id_log + \
              pm_log + gdm_log + ldm_log + \
              total_loss_log
        print(log)

    #############
    # Epoch Log #
    #############

    time_log = 'Ep {}, {:.2f}s'.format(ep + 1, time.time() - ep_st, )

    if cfg.g_loss_weight > 0:
      g_log = (', gp {:.2%}, gm {:.2%}, '
               'gd_ap {:.4f}, gd_an {:.4f}, '
               'gL {:.4f}'.format(
        g_prec_meter.avg, g_m_meter.avg,
        g_dist_ap_meter.avg, g_dist_an_meter.avg,
        g_loss_meter.avg, ))
    else:
      g_log = ''

    if cfg.l_loss_weight > 0:
      l_log = (', lp {:.2%}, lm {:.2%}, '
               'ld_ap {:.4f}, ld_an {:.4f}, '
               'lL {:.4f}'.format(
        l_prec_meter.avg, l_m_meter.avg,
        l_dist_ap_meter.avg, l_dist_an_meter.avg,
        l_loss_meter.avg, ))
    else:
      l_log = ''

    if cfg.id_loss_weight > 0:
      id_log = (', idL {:.4f}'.format(id_loss_meter.avg))
    else:
      id_log = ''

    if (cfg.num_models > 1) and (cfg.pm_loss_weight > 0):
      pm_log = (', pmL {:.4f}'.format(pm_loss_meter.avg))
    else:
      pm_log = ''

    if (cfg.num_models > 1) and (cfg.gdm_loss_weight > 0):
      gdm_log = (', gdmL {:.4f}'.format(gdm_loss_meter.avg))
    else:
      gdm_log = ''

    if (cfg.num_models > 1) \
        and cfg.local_dist_own_hard_sample \
        and (cfg.ldm_loss_weight > 0):
      ldm_log = (', ldmL {:.4f}'.format(ldm_loss_meter.avg))
    else:
      ldm_log = ''

    total_loss_log = ', loss {:.4f}'.format(loss_meter.avg)

    log = time_log + \
          g_log + l_log + id_log + \
          pm_log + gdm_log + ldm_log + \
          total_loss_log
    print(log)

    # Log to TensorBoard

    if cfg.log_to_file:
      if writer is None:
        writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
      writer.add_scalars(
        'loss',
        dict(global_loss=g_loss_meter.avg,
             local_loss=l_loss_meter.avg,
             id_loss=id_loss_meter.avg,
             pm_loss=pm_loss_meter.avg,
             gdm_loss=gdm_loss_meter.avg,
             ldm_loss=ldm_loss_meter.avg,
             loss=loss_meter.avg, ),
        ep)
      writer.add_scalars(
        'tri_precision',
        dict(global_precision=g_prec_meter.avg,
             local_precision=l_prec_meter.avg, ),
        ep)
      writer.add_scalars(
        'satisfy_margin',
        dict(global_satisfy_margin=g_m_meter.avg,
             local_satisfy_margin=l_m_meter.avg, ),
        ep)
      writer.add_scalars(
        'global_dist',
        dict(global_dist_ap=g_dist_ap_meter.avg,
             global_dist_an=g_dist_an_meter.avg, ),
        ep)
      writer.add_scalars(
        'local_dist',
        dict(local_dist_ap=l_dist_ap_meter.avg,
             local_dist_an=l_dist_an_meter.avg, ),
        ep)

    # save ckpt
    if cfg.log_to_file:
      save_ckpt(modules_optims, ep + 1, 0, cfg.ckpt_file)

  ########
  # Test #
  ########

  test(load_model_weight=False)


if __name__ == '__main__':
  main()
