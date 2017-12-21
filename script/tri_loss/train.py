from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

import time
import shutil
import os.path as osp
from tensorboardX import SummaryWriter
import threading

from train_cfg import Config

from aligned_reid.tri_loss.dataset import create_dataset
from aligned_reid.tri_loss.model.Model import Model
from aligned_reid.tri_loss.model.TripletLoss import TripletLoss
from aligned_reid.tri_loss.model.loss import global_loss
from aligned_reid.tri_loss.model.loss import local_loss

from aligned_reid.utils.utils import may_set_mode
from aligned_reid.utils.utils import load_ckpt
from aligned_reid.utils.utils import save_ckpt
from aligned_reid.utils.utils import set_devices_for_ml
from aligned_reid.utils.utils import AverageMeter
from aligned_reid.utils.utils import to_scalar
from aligned_reid.utils.utils import ReDirectSTD
from aligned_reid.utils.utils import set_seed
from aligned_reid.utils.utils import print_array
from aligned_reid.utils.utils import find_index


def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
  """Decay exponentially after some epochs."""
  if ep > start_decay_at_ep:
    for g in optimizer.param_groups:
      g['lr'] = (base_lr * (0.001 ** (float(ep - start_decay_at_ep)
                                      / (total_ep - start_decay_at_ep))))
    print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


def adjust_lr_staircase(optimizer, base_lr, ep, decay_at_epochs, factor):
  """Multiplied by a factor at specified epochs."""
  if ep in decay_at_epochs:
    ind = find_index(decay_at_epochs, ep)
    for g in optimizer.param_groups:
      g['lr'] = base_lr * factor ** (ind + 1)
    print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


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
    ReDirectSTD(cfg.log_file, 'stdout', False)
    ReDirectSTD(cfg.log_err_file, 'stderr', False)

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
  # Models  #
  ###########

  models = [Model(local_conv_out_channels=cfg.local_conv_out_channels,
                  num_classes=cfg.num_classes)
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

  ###########
  # Dataset #
  ###########

  if not cfg.only_test:
    train_set = create_dataset(**cfg.train_set_kwargs)
  test_set = create_dataset(**cfg.test_set_kwargs)

  ########
  # Test #
  ########

  # Test each model using different distance settings.
  def test(load_from_ckpt=False):
    if load_from_ckpt:
      load_ckpt(modules_optims, cfg.ckpt_file)

    for i in range(cfg.num_models):
      test_set.set_feat_func(ExtractFeature(model_ws[i], TVTs[i]))

      print('=====> Test Model {}'.format(i + 1))
      use_local_distance = (cfg.l_loss_weight > 0) \
                           and cfg.local_dist_own_hard_sample
      test_set.eval(
        normalize_feat=cfg.normalize_feature,
        use_local_distance=use_local_distance)

  if cfg.only_test:
    test(load_from_ckpt=True)
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
      probs = F.softmax(logits)
      log_probs = F.log_softmax(logits)

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

      # These meters are outer-scope variables

      # Just record for the first model
      if i == 0:

        # precision
        g_prec = (g_dist_an > g_dist_ap).data.float().mean()
        # the proportion of triplets that satisfy margin
        g_m = (g_dist_an > g_dist_ap + cfg.global_margin).data.float().mean()
        g_d_ap = g_dist_ap.data.mean()
        g_d_an = g_dist_an.data.mean()

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
          ep,
          cfg.total_epochs,
          cfg.exp_decay_at_epoch)
      else:
        adjust_lr_staircase(
          optimizer,
          cfg.base_lr,
          ep,
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
          g_log = (', gp {:.4f}, gm {:.4f}, '
                   'gd_ap {:.4f}, gd_an {:.4f}, '
                   'g_loss {:.4f}'.format(
            g_prec_meter.val, g_m_meter.val,
            g_dist_ap_meter.val, g_dist_an_meter.val,
            g_loss_meter.val, ))
        else:
          g_log = ''

        if cfg.l_loss_weight > 0:
          l_log = (', lp {:.4f}, lm {:.4f}, '
                   'ld_ap {:.4f}, ld_an {:.4f}, '
                   'l_loss {:.4f}'.format(
            l_prec_meter.val, l_m_meter.val,
            l_dist_ap_meter.val, l_dist_an_meter.val,
            l_loss_meter.val, ))
        else:
          l_log = ''

        if cfg.id_loss_weight > 0:
          id_log = (', id_loss {:.4f}'.format(id_loss_meter.val))
        else:
          id_log = ''

        if (cfg.num_models > 1) and (cfg.pm_loss_weight > 0):
          pm_log = (', pm_loss {:.4f}'.format(pm_loss_meter.val))
        else:
          pm_log = ''

        if (cfg.num_models > 1) and (cfg.gdm_loss_weight > 0):
          gdm_log = (', gdm_loss {:.4f}'.format(gdm_loss_meter.val))
        else:
          gdm_log = ''

        if (cfg.num_models > 1) \
            and cfg.local_dist_own_hard_sample \
            and (cfg.ldm_loss_weight > 0):
          ldm_log = (', ldm_loss {:.4f}'.format(ldm_loss_meter.val))
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
      g_log = (', gp {:.4f}, gm {:.4f}, '
               'gd_ap {:.4f}, gd_an {:.4f}, '
               'g_loss {:.4f}'.format(
        g_prec_meter.avg, g_m_meter.avg,
        g_dist_ap_meter.avg, g_dist_an_meter.avg,
        g_loss_meter.avg, ))
    else:
      g_log = ''

    if cfg.l_loss_weight > 0:
      l_log = (', lp {:.4f}, lm {:.4f}, '
               'ld_ap {:.4f}, ld_an {:.4f}, '
               'l_loss {:.4f}'.format(
        l_prec_meter.avg, l_m_meter.avg,
        l_dist_ap_meter.avg, l_dist_an_meter.avg,
        l_loss_meter.avg, ))
    else:
      l_log = ''

    if cfg.id_loss_weight > 0:
      id_log = (', id_loss {:.4f}'.format(id_loss_meter.avg))
    else:
      id_log = ''

    if (cfg.num_models > 1) and (cfg.pm_loss_weight > 0):
      pm_log = (', pm_loss {:.4f}'.format(pm_loss_meter.avg))
    else:
      pm_log = ''

    if (cfg.num_models > 1) and (cfg.gdm_loss_weight > 0):
      gdm_log = (', gdm_loss {:.4f}'.format(gdm_loss_meter.avg))
    else:
      gdm_log = ''

    if (cfg.num_models > 1) \
        and cfg.local_dist_own_hard_sample \
        and (cfg.ldm_loss_weight > 0):
      ldm_log = (', ldm_loss {:.4f}'.format(ldm_loss_meter.avg))
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

  test(load_from_ckpt=False)


if __name__ == '__main__':
  main()
