from __future__ import print_function
from .Dataset import Dataset
from aligned_reid.utils.ranking import cmc, mean_ap
from aligned_reid.utils.dataset_utils import parse_im_name
from aligned_reid.utils.distance import normalize
from aligned_reid.utils.distance import compute_dist
from aligned_reid.utils.distance import local_dist

import time
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class TestSet(Dataset):
  """Test set for triplet loss.
  Args:
    extract_feat: a function to extract features. It takes a batch of images 
      and returns a batch of features.
    marks: a list, each element e denoting whether the image is from 
      query (e == 0), or
      gallery (e == 1), or 
      multi query (e == 2) set
  """

  def __init__(self,
               im_dir=None,
               im_names=None,
               marks=None,
               extract_feat_func=None,
               separate_camera_set=None,
               single_gallery_shot=None,
               first_match_break=None,
               **kwargs):
    super(TestSet, self).__init__(dataset_size=len(im_names), **kwargs)
    # The im dir of all images
    self.im_dir = im_dir
    self.im_names = im_names
    self.marks = marks
    self.extract_feat_func = extract_feat_func
    self.separate_camera_set = separate_camera_set
    self.single_gallery_shot = single_gallery_shot
    self.first_match_break = first_match_break

  def set_feat_func(self, extract_feat_func):
    self.extract_feat_func = extract_feat_func

  def get_sample(self, ptr):
    im_name = self.im_names[ptr]
    im_path = osp.join(self.im_dir, im_name)
    im = plt.imread(im_path)
    im, _ = self.pre_process_im(im)
    id = parse_im_name(self.im_names[ptr], 'id')
    cam = parse_im_name(self.im_names[ptr], 'cam')
    # denoting whether the im is from query, gallery, or multi query set
    mark = self.marks[ptr]
    return im, id, cam, im_name, mark

  def next_batch(self):
    if self.epoch_done and self.shuffle:
      self.prng.shuffle(self.im_names)
    samples, self.epoch_done = self.prefetcher.next_batch()
    im_list, ids, cams, im_names, marks = zip(*samples)
    # Transform the list into a numpy array with shape [N, ...]
    ims = np.stack(im_list, axis=0)
    ids = np.array(ids)
    cams = np.array(cams)
    im_names = np.array(im_names)
    marks = np.array(marks)
    return ims, ids, cams, im_names, marks, self.epoch_done

  def extract_feat(self, normalize_feat):
    """Extract the features of the whole image set.
    Args:
      normalize_feat: True or False, whether to normalize global and local 
        feature to unit length
    Returns:
      global_feats: numpy array with shape [N, C]
      local_feats: numpy array with shape [N, H, c]
      ids: numpy array with shape [N]
      cams: numpy array with shape [N]
      im_names: numpy array with shape [N]
      marks: numpy array with shape [N]
    """
    global_feats, local_feats, ids, cams, im_names, marks = \
      [], [], [], [], [], []
    done = False
    step = 0
    last_time = time.time()
    while not done:
      ims_, ids_, cams_, im_names_, marks_, done = self.next_batch()
      global_feat, local_feat = self.extract_feat_func(ims_)
      global_feats.append(global_feat)
      local_feats.append(local_feat)
      ids.append(ids_)
      cams.append(cams_)
      im_names.append(im_names_)
      marks.append(marks_)

      # log
      total_batches = (self.prefetcher.dataset_size
                       // self.prefetcher.batch_size + 1)
      step += 1
      if step % 50 == 0:
        print('\t{}/{} batches done, +{:.2f}s'
              .format(step, total_batches, time.time() - last_time))
        last_time = time.time()

    global_feats = np.vstack(global_feats)
    local_feats = np.concatenate(local_feats)
    ids = np.hstack(ids)
    cams = np.hstack(cams)
    im_names = np.hstack(im_names)
    marks = np.hstack(marks)
    if normalize_feat:
      global_feats = normalize(global_feats, axis=1)
      local_feats = normalize(local_feats, axis=-1)
    return global_feats, local_feats, ids, cams, im_names, marks

  @staticmethod
  def eval_map_cmc(
      q_g_dist,
      q_ids=None, g_ids=None,
      q_cams=None, g_cams=None,
      separate_camera_set=None,
      single_gallery_shot=None,
      first_match_break=None,
      topk=None):
    """Compute CMC and mAP.
    Args:
      q_g_dist: numpy array with shape [num_query, num_gallery], the 
        pairwise distance between query and gallery samples
    Returns:
      mAP: numpy array with shape [num_query], the AP averaged across query 
        samples
      cmc_scores: numpy array with shape [topk], the cmc curve 
        averaged across query samples
    """
    # Compute mean AP
    mAP = mean_ap(
      distmat=q_g_dist,
      query_ids=q_ids, gallery_ids=g_ids,
      query_cams=q_cams, gallery_cams=g_cams)
    # Compute CMC scores
    cmc_scores = cmc(
      distmat=q_g_dist,
      query_ids=q_ids, gallery_ids=g_ids,
      query_cams=q_cams, gallery_cams=g_cams,
      separate_camera_set=separate_camera_set,
      single_gallery_shot=single_gallery_shot,
      first_match_break=first_match_break,
      topk=topk)
    print('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]'
          .format(mAP, *cmc_scores[[0, 4, 9]]))
    return mAP, cmc_scores

  def eval(
      self,
      normalize_feat=True,
      global_weight=1.,
      local_weight=0.,
      pool_type='average'):
    """Evaluate using metric CMC and mAP.
    Args:
      normalize_feat: whether to normalize features before computing distance
      global_weight: weight of global distance
      local_weight: weight of local distance
      pool_type: 'average' or 'max', only for multi-query case
    """
    st = time.time()
    print('Extracting feature...')
    global_feats, local_feats, ids, cams, im_names, marks = \
      self.extract_feat(normalize_feat)
    print('Done, {:.2f}s'.format(time.time() - st))
    q_inds = marks == 0
    g_inds = marks == 1
    mq_inds = marks == 2

    q_g_dist = 0

    if global_weight > 0:
      st = time.time()
      print('Computing global distance...')
      global_q_g_dist = compute_dist(
        global_feats[q_inds], global_feats[g_inds], type='euclidean')
      print('Done, {:.2f}s'.format(time.time() - st))
      q_g_dist += global_q_g_dist

    if local_weight > 0:
      st = time.time()
      last_time = time.time()
      print('Computing local distance...')
      q_local_feats = local_feats[q_inds]
      g_local_feats = local_feats[g_inds]
      # In order not to flood the memory with huge data,
      # split the gallery set into smaller parts (Divide and Conquer).
      # Even if memory may be enough to store the large matrix, frequently
      # allocating and freeing large memory (e.g. dozens of GB) alone takes
      # MUCH time.
      num_splits = int(len(g_local_feats) / 100) + 1
      local_q_g_dist = []
      for i, glf in enumerate(np.array_split(g_local_feats, num_splits)):
        local_q_g_dist.append(local_dist(q_local_feats, glf))
        print('\tsplit {}/{}, +{:.2f}s'
              .format(i + 1, num_splits, time.time() - last_time))
        last_time = time.time()
      local_q_g_dist = np.concatenate(local_q_g_dist, axis=1)
      print('Done, {:.2f}s'.format(time.time() - st))

      q_g_dist += local_q_g_dist

    st = time.time()
    print('Computing scores...')
    mAP, cmc_scores = self.eval_map_cmc(
      q_g_dist=q_g_dist,
      q_ids=ids[q_inds], g_ids=ids[g_inds],
      q_cams=cams[q_inds], g_cams=cams[g_inds],
      separate_camera_set=self.separate_camera_set,
      single_gallery_shot=self.single_gallery_shot,
      first_match_break=self.first_match_break,
      topk=10)
    print('Done, {:.2f}s'.format(time.time() - st))

    # multi-query
    # TODO: allow local distance in Multi Query
    mq_mAP, mq_cmc_scores = None, None

    return mAP, cmc_scores, mq_mAP, mq_cmc_scores
