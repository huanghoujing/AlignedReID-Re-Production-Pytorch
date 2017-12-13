from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import time
import shutil
import os.path as osp
from tensorboardX import SummaryWriter

from train_cfg import Config

from aligned_reid.tri_loss.dataset import create_dataset
from aligned_reid.tri_loss.model.Model import Model
from aligned_reid.tri_loss.model.TripletLoss import TripletLoss
from aligned_reid.tri_loss.model.loss import global_loss
from aligned_reid.tri_loss.model.loss import local_loss

from aligned_reid.utils.utils import may_set_mode
from aligned_reid.utils.utils import load_ckpt
from aligned_reid.utils.utils import save_ckpt
from aligned_reid.utils.utils import set_devices
# from aligned_reid.utils.utils import adjust_lr
from aligned_reid.utils.utils import AverageMeter
from aligned_reid.utils.utils import to_scalar
from aligned_reid.utils.utils import ReDirectSTD
from aligned_reid.utils.utils import set_seed
from aligned_reid.utils.utils import get_model_wrapper
from aligned_reid.utils.utils import print_array


def adjust_lr(optimizer, base_lr, ep, total_ep, tran_ep):
  if ep > tran_ep:
    for g in optimizer.param_groups:
      g['lr'] = (base_lr *
                 (0.001 ** (float(ep - tran_ep) / (total_ep - tran_ep))))
      print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


def main():
  cfg = Config()

  # Redirect logs to both console and file.
  if cfg.log_to_file:
    ReDirectSTD(cfg.log_file, 'stdout', False)
    ReDirectSTD(cfg.log_err_file, 'stderr', False)

  TVT, TMO = set_devices(cfg.sys_device_ids)
  if cfg.seed is not None:
    set_seed(cfg.seed)

  # Dump the configurations to log.
  import pprint
  pprint.pprint(cfg.__dict__)

  if cfg.log_to_file:
    writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
  else:
    writer = None

  ###########
  # Models  #
  ###########

  model = Model(local_conv_out_channels=cfg.local_conv_out_channels,
                num_classes=cfg.num_classes)
  model_w = get_model_wrapper(model, len(cfg.sys_device_ids) > 1)

  #############################
  # Criteria and Optimizers   #
  #############################

  id_criterion = nn.CrossEntropyLoss()
  g_tri_loss = TripletLoss(margin=cfg.global_margin)
  l_tri_loss = TripletLoss(margin=cfg.local_margin)

  optimizer = optim.Adam(model.parameters(),
                         lr=cfg.lr,
                         weight_decay=cfg.weight_decay)

  modules_optims = [model, optimizer]

  ################################
  # May Resume Models and Optims #
  ################################

  if cfg.resume:
    resume_ep, scores = load_ckpt(modules_optims, cfg.ckpt_file)

  # May Transfer Models and Optims to Specified Device
  TMO(modules_optims)

  ###########
  # Dataset #
  ###########

  def feature_func(ims):
    """A function to be called in the val/test set, to extract features."""
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    may_set_mode(modules_optims, 'eval')
    ims = Variable(TVT(torch.from_numpy(ims).float()))
    global_feat, local_feat = model_w(ims)[:2]
    global_feat = global_feat.data.cpu().numpy()
    local_feat = local_feat.data.cpu().numpy()
    return global_feat, local_feat

  train_set, val_set, test_set = None, None, None
  if not cfg.only_test:
    train_set = create_dataset(**cfg.train_set_kwargs)
    # val_set = create_dataset(**cfg.val_set_kwargs)
    # val_set.set_feat_func(feature_func)
  if cfg.only_test or cfg.test:
    test_set = create_dataset(**cfg.test_set_kwargs)
    test_set.set_feat_func(feature_func)

  ########
  # Test #
  ########

  if cfg.only_test:
    print('=====> Test')
    load_ckpt(modules_optims, cfg.ckpt_file)
    mAP, cmc_scores, mq_mAP, mq_cmc_scores = test_set.eval(
      normalize_feat=cfg.normalize_feature,
      global_weight=cfg.g_test_weight,
      local_weight=cfg.l_test_weight)
    return

  ############
  # Training #
  ############

  best_score = scores if cfg.resume else 0
  start_ep = resume_ep if cfg.resume else 0
  for ep in range(start_ep, cfg.num_epochs):
    adjust_lr(optimizer, cfg.lr, ep, cfg.num_epochs, cfg.start_decay_epoch)
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

    loss_meter = AverageMeter()

    ep_st = time.time()
    step = 0
    while not epoch_done:

      step += 1
      step_st = time.time()

      ims, im_names, labels, mirrored, epoch_done = train_set.next_batch()

      ims_var = Variable(TVT(torch.from_numpy(ims).float()))
      labels_t = TVT(torch.from_numpy(labels).long())
      labels_var = Variable(labels_t)
      global_feat, local_feat, logits = model_w(ims_var)

      g_loss, p_inds, n_inds, g_dist_ap, g_dist_an = global_loss(
        g_tri_loss, global_feat, labels_t,
        normalize_feature=cfg.normalize_feature)

      if cfg.l_loss_weight == 0:
        l_loss, l_prec, l_m = 0, 0, 0
      elif cfg.local_dist_own_hard_sample:
        # Let local distance find its own hard samples.
        l_loss, l_dist_ap, l_dist_an = local_loss(
          l_tri_loss, local_feat, None, None, labels_t,
          normalize_feature=cfg.normalize_feature)
      else:
        l_loss, l_dist_ap, l_dist_an = local_loss(
          l_tri_loss, local_feat, p_inds, n_inds, labels_t,
          normalize_feature=cfg.normalize_feature)

      if cfg.id_loss_weight == 0:
        id_loss = 0
      else:
        id_loss = id_criterion(logits, labels_var)

      loss = g_loss * cfg.g_loss_weight \
             + l_loss * cfg.l_loss_weight \
             + id_loss * cfg.id_loss_weight

      optimizer.zero_grad()
      loss.backward()

      optimizer.step()

      ############
      # Step Log #
      ############

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

      loss_meter.update(to_scalar(loss))

      if step % cfg.log_steps == 0:
        time_log = '\tStep {}/Ep {}, {:.2f}s'.format(
          step, ep + 1, time.time() - step_st,)

        if cfg.g_loss_weight > 0:
          g_log = (', gp {:.4f}, gm {:.4f}, '
                   'gd_ap {:.4f}, gd_an {:.4f}, '
                   'g_loss {:.4f}'.format(
            g_prec_meter.val, g_m_meter.val,
            g_dist_ap_meter.val, g_dist_an_meter.val,
            g_loss_meter.val,))
        else:
          g_log = ''

        if cfg.l_loss_weight > 0:
          l_log = (', lp {:.4f}, lm {:.4f}, '
                   'ld_ap {:.4f}, ld_an {:.4f}, '
                   'l_loss {:.4f}'.format(
            l_prec_meter.val, l_m_meter.val,
            l_dist_ap_meter.val, l_dist_an_meter.val,
            l_loss_meter.val,))
        else:
          l_log = ''

        if cfg.id_loss_weight > 0:
          id_log = (', id_loss {:.4f}'.format(id_loss_meter.val))
        else:
          id_log = ''

        total_loss_log = ', loss {:.4f}'.format(loss_meter.val)

        log = time_log + g_log + l_log + id_log + total_loss_log
        print(log)

    #############
    # Epoch Log #
    #############

    time_log = 'Ep {}, {:.2f}s'.format(ep + 1, time.time() - ep_st,)

    if cfg.g_loss_weight > 0:
      g_log = (', gp {:.4f}, gm {:.4f}, '
               'gd_ap {:.4f}, gd_an {:.4f}, '
               'g_loss {:.4f}'.format(
        g_prec_meter.avg, g_m_meter.avg,
        g_dist_ap_meter.avg, g_dist_an_meter.avg,
        g_loss_meter.avg,))
    else:
      g_log = ''

    if cfg.l_loss_weight > 0:
      l_log = (', lp {:.4f}, lm {:.4f}, '
               'ld_ap {:.4f}, ld_an {:.4f}, '
               'l_loss {:.4f}'.format(
        l_prec_meter.avg, l_m_meter.avg,
        l_dist_ap_meter.avg, l_dist_an_meter.avg,
        l_loss_meter.avg,))
    else:
      l_log = ''

    if cfg.id_loss_weight > 0:
      id_log = (', id_loss {:.4f}'.format(id_loss_meter.avg))
    else:
      id_log = ''

    total_loss_log = ', loss {:.4f}'.format(loss_meter.avg)

    log = time_log + g_log + l_log + id_log + total_loss_log
    print(log)

    # Log to TensorBoard

    if cfg.log_to_file:
      writer.add_scalars(
        'loss',
        dict(global_loss=g_loss_meter.avg,
             local_loss=l_loss_meter.avg,
             id_loss=id_loss_meter.avg,
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

    mAP = 0
    # print('=====> Validation')
    # mAP, cmc_scores, mq_mAP, mq_cmc_scores = val_set.eval(
    #   normalize_feat=cfg.normalize_feature,
    #   global_weight=cfg.g_test_weight,
    #   local_weight=cfg.l_test_weight)

    # save ckpt
    if cfg.log_to_file:
      save_ckpt(modules_optims, ep + 1, mAP, cfg.ckpt_file)
      # if mAP > best_score:
      #   best_score = mAP
      #   shutil.copy(cfg.ckpt_file, cfg.best_ckpt_file)

  ########
  # Test #
  ########

  if cfg.test:
    print('=====> Test')
    mAP, cmc_scores, mq_mAP, mq_cmc_scores = test_set.eval(
      normalize_feat=cfg.normalize_feature,
      global_weight=cfg.g_test_weight,
      local_weight=cfg.l_test_weight)


if __name__ == '__main__':
  main()
