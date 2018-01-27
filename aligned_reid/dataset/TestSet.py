from __future__ import print_function
import sys
import time
import os.path as osp
from PIL import Image
import numpy as np

from .Dataset import Dataset

from ..utils.utils import measure_time
from ..utils.re_ranking import re_ranking
from ..utils.metric import cmc, mean_ap
from ..utils.dataset_utils import parse_im_name
from ..utils.distance import normalize
from ..utils.distance import compute_dist
from ..utils.distance import local_dist
from ..utils.distance import low_memory_matrix_op


class TestSet(Dataset):
  """
  Args:
    extract_feat_func: a function to extract features. It takes a batch of
      images and returns a batch of features.
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
    im = np.asarray(Image.open(im_path))
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
    st = time.time()
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
        print('{}/{} batches done, +{:.2f}s, total {:.2f}s'
              .format(step, total_batches,
                      time.time() - last_time, time.time() - st))
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
    with measure_time('Extracting feature...'):
      global_feats, local_feats, ids, cams, im_names, marks = \
        self.extract_feat(normalize_feat)

    # query, gallery, multi-query indices
    q_inds = marks == 0
    g_inds = marks == 1
    mq_inds = marks == 2

    # A helper function just for avoiding code duplication.
    def compute_score(dist_mat):
      mAP, cmc_scores = self.eval_map_cmc(
        q_g_dist=dist_mat,
        q_ids=ids[q_inds], g_ids=ids[g_inds],
        q_cams=cams[q_inds], g_cams=cams[g_inds],
        separate_camera_set=self.separate_camera_set,
        single_gallery_shot=self.single_gallery_shot,
        first_match_break=self.first_match_break,
        topk=10)
      return mAP, cmc_scores

    # A helper function just for avoiding code duplication.
    def low_memory_local_dist(x, y):
      with measure_time('Computing local distance...'):
        x_num_splits = int(len(x) / 200) + 1
        y_num_splits = int(len(y) / 200) + 1
        z = low_memory_matrix_op(
          local_dist, x, y, 0, 0, x_num_splits, y_num_splits, verbose=True)
      return z

    ###################
    # Global Distance #
    ###################

    with measure_time('Computing global distance...'):
      # query-gallery distance using global distance
      global_q_g_dist = compute_dist(
        global_feats[q_inds], global_feats[g_inds], type='euclidean')

    with measure_time('Computing scores for Global Distance...'):
      mAP, cmc_scores = compute_score(global_q_g_dist)

    if to_re_rank:
      with measure_time('Re-ranking...'):
        # query-query distance using global distance
        global_q_q_dist = compute_dist(
          global_feats[q_inds], global_feats[q_inds], type='euclidean')

        # gallery-gallery distance using global distance
        global_g_g_dist = compute_dist(
          global_feats[g_inds], global_feats[g_inds], type='euclidean')

        # re-ranked global query-gallery distance
        re_r_global_q_g_dist = re_ranking(
          global_q_g_dist, global_q_q_dist, global_g_g_dist)

      with measure_time('Computing scores for re-ranked Global Distance...'):
        mAP, cmc_scores = compute_score(re_r_global_q_g_dist)


    if use_local_distance:

      ##################
      # Local Distance #
      ##################

      # query-gallery distance using local distance
      local_q_g_dist = low_memory_local_dist(
        local_feats[q_inds], local_feats[g_inds])

      with measure_time('Computing scores for Local Distance...'):
        mAP, cmc_scores = compute_score(local_q_g_dist)

      if to_re_rank:
        with measure_time('Re-ranking...'):
          # query-query distance using local distance
          local_q_q_dist = low_memory_local_dist(
            local_feats[q_inds], local_feats[q_inds])

          # gallery-gallery distance using local distance
          local_g_g_dist = low_memory_local_dist(
            local_feats[g_inds], local_feats[g_inds])

          re_r_local_q_g_dist = re_ranking(
            local_q_g_dist, local_q_q_dist, local_g_g_dist)

        with measure_time('Computing scores for re-ranked Local Distance...'):
          mAP, cmc_scores = compute_score(re_r_local_q_g_dist)

      #########################
      # Global+Local Distance #
      #########################

      global_local_q_g_dist = global_q_g_dist + local_q_g_dist
      with measure_time('Computing scores for Global+Local Distance...'):
        mAP, cmc_scores = compute_score(global_local_q_g_dist)

      if to_re_rank:
        with measure_time('Re-ranking...'):
          global_local_q_q_dist = global_q_q_dist + local_q_q_dist
          global_local_g_g_dist = global_g_g_dist + local_g_g_dist

          re_r_global_local_q_g_dist = re_ranking(
            global_local_q_g_dist, global_local_q_q_dist, global_local_g_g_dist)

        with measure_time(
            'Computing scores for re-ranked Global+Local Distance...'):
          mAP, cmc_scores = compute_score(re_r_global_local_q_g_dist)


    # multi-query
    # TODO: allow local distance in Multi Query
    mq_mAP, mq_cmc_scores = None, None

    return mAP, cmc_scores, mq_mAP, mq_cmc_scores
