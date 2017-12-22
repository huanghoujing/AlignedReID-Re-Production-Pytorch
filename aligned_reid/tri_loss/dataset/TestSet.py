from __future__ import print_function
import sys
import time
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from .Dataset import Dataset

from aligned_reid.utils.re_ranking import re_ranking
from aligned_reid.utils.metric import cmc, mean_ap
from aligned_reid.utils.dataset_utils import parse_im_name
from aligned_reid.utils.distance import normalize
from aligned_reid.utils.distance import compute_dist
from aligned_reid.utils.distance import local_dist
from aligned_reid.utils.distance import low_memory_matrix_op


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

  def __init__(
      self,
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
    printed = False
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
      if step % 20 == 0:
        if not printed:
          printed = True
        else:
          # Clean the current line
          sys.stdout.write("\033[F\033[K")
        print('{}/{} batches done, +{:.2f}s'
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
      use_local_distance=False,
      to_re_rank=True,
      pool_type='average'):
    """Evaluate using metric CMC and mAP.
    Args:
      normalize_feat: whether to normalize features before computing distance
      use_local_distance: whether to use local distance
      to_re_rank: whether to also report re-ranking scores
      pool_type: 'average' or 'max', only for multi-query case
    """
    st = time.time()
    print('Extracting feature...')
    global_feats, local_feats, ids, cams, im_names, marks = \
      self.extract_feat(normalize_feat)
    print('Done, {:.2f}s'.format(time.time() - st))

    # query, gallery, multi-query indices
    q_inds = marks == 0
    g_inds = marks == 1
    mq_inds = marks == 2

    ###################
    # Global Distance #
    ###################

    st = time.time()
    print('Computing global distance...')
    # query-gallery distance using global distance
    global_q_g_dist = compute_dist(
      global_feats[q_inds], global_feats[g_inds], type='euclidean')
    print('Done, {:.2f}s'.format(time.time() - st))

    ##################
    # Local Distance #
    ##################

    if use_local_distance:
      st = time.time()
      print('Computing local distance...')
      q_local_feats = local_feats[q_inds]
      g_local_feats = local_feats[g_inds]

      # query-gallery distance using local distance
      num_splits = int(len(g_local_feats) / 50) + 1
      local_q_g_dist = low_memory_matrix_op(
        q_local_feats, g_local_feats, local_dist, 'y', 0, num_splits,
        verbose=True)

      print('Done, {:.2f}s'.format(time.time() - st))

    # A helper function just for avoiding code duplication.
    def compute_score(dist_mat, dist_type):
      st = time.time()
      print('Computing scores for {}...'.format(dist_type))
      mAP, cmc_scores = self.eval_map_cmc(
        q_g_dist=dist_mat,
        q_ids=ids[q_inds], g_ids=ids[g_inds],
        q_cams=cams[q_inds], g_cams=cams[g_inds],
        separate_camera_set=self.separate_camera_set,
        single_gallery_shot=self.single_gallery_shot,
        first_match_break=self.first_match_break,
        topk=10)
      print('{} score done, {:.2f}s\n'
            .format(dist_type, time.time() - st))
      return mAP, cmc_scores

    ##################################
    # Compute Global Distance Scores #
    ##################################

    mAP, cmc_scores = compute_score(global_q_g_dist, 'Global Distance')

    if to_re_rank:
      rr_st = time.time()
      print('Re-ranking...')

      # query-query distance using global distance
      global_q_q_dist = compute_dist(
        global_feats[q_inds], global_feats[q_inds], type='euclidean')
      # gallery-gallery distance using global distance
      global_g_g_dist = compute_dist(
        global_feats[g_inds], global_feats[g_inds], type='euclidean')

      # re-ranked global query-gallery distance
      re_r_global_q_g_dist = re_ranking(
        global_q_g_dist, global_q_q_dist, global_g_g_dist)
      print('Re-ranking done, {:.2f}s'.format(time.time() - rr_st))

      mAP, cmc_scores = compute_score(
        re_r_global_q_g_dist, 're-ranked Global Distance')

    ##################################
    # Compute Local Distance Scores #
    ##################################

    if use_local_distance:
      mAP, cmc_scores = compute_score(local_q_g_dist, 'Local Distance')

      if to_re_rank:
        rr_st = time.time()
        print('Re-ranking...')

        num_splits = int(len(q_local_feats) / 50) + 1
        # query-query distance using local distance
        local_q_q_dist = low_memory_matrix_op(
          q_local_feats, q_local_feats, local_dist, 'y', 0, num_splits,
          verbose=True)

        num_splits = int(len(g_local_feats) / 50) + 1
        # gallery-gallery distance using local distance
        local_g_g_dist = low_memory_matrix_op(
          g_local_feats, g_local_feats, local_dist, 'y', 0, num_splits,
          verbose=True)

        re_r_local_q_g_dist = re_ranking(
          local_q_g_dist, local_q_q_dist, local_g_g_dist)

        print('Re-ranking done, {:.2f}s'.format(time.time() - rr_st))

        mAP, cmc_scores = compute_score(
          re_r_local_q_g_dist, 're-ranked Local Distance')

      ########################################
      # Compute Global+Local Distance Scores #
      ########################################

      global_local_q_g_dist = global_q_g_dist + local_q_g_dist
      mAP, cmc_scores = compute_score(
          global_local_q_g_dist, 'Global+Local Distance')

      if to_re_rank:
        rr_st = time.time()
        print('Re-ranking...')

        global_local_q_q_dist = global_q_q_dist + local_q_q_dist
        global_local_g_g_dist = global_g_g_dist + local_g_g_dist

        re_r_global_local_q_g_dist = re_ranking(
          global_local_q_g_dist, global_local_q_q_dist, global_local_g_g_dist)

        print('Re-ranking done, {:.2f}s'.format(time.time() - rr_st))

        mAP, cmc_scores = compute_score(
          re_r_global_local_q_g_dist, 're-ranked Global+Local Distance')


    # multi-query
    # TODO: allow local distance in Multi Query
    mq_mAP, mq_cmc_scores = None, None

    return mAP, cmc_scores, mq_mAP, mq_cmc_scores
