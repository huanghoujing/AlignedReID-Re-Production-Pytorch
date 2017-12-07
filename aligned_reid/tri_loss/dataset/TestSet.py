from __future__ import print_function
from .Dataset import Dataset
from aligned_reid.utils.ranking import cmc, mean_ap
from aligned_reid.utils.dataset_utils import parse_im_name
from aligned_reid.utils.dataset_utils import normalize
from aligned_reid.utils.dataset_utils import compute_dist

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
      normalize_feat: True or False, whether to normalize each image feature to 
        vector with unit-length
    Returns:
      feats: numpy array with shape [N, feat_dim]
      ids: numpy array with shape [N]
      cams: numpy array with shape [N]
      im_names: numpy array with shape [N]
      marks: numpy array with shape [N]
    """
    feats, ids, cams, im_names, marks = [], [], [], [], []
    done = False
    while not done:
      ims_, ids_, cams_, im_names_, marks_, done = self.next_batch()
      feats.append(self.extract_feat_func(ims_))
      ids.append(ids_)
      cams.append(cams_)
      im_names.append(im_names_)
      marks.append(marks_)
    feats = np.vstack(feats)
    ids = np.hstack(ids)
    cams = np.hstack(cams)
    im_names = np.hstack(im_names)
    marks = np.hstack(marks)
    if normalize_feat:
      feats = normalize(feats, axis=1)
    return feats, ids, cams, im_names, marks

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
      cmc_scores: numpy array (with shape [num_query-1]?), the cmc curve 
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

  def eval(self, normalize_feat=True, pool_type='average'):
    """Evaluate using metric CMC and mAP.
    Args:
      normalize_feat: whether to normalize features before computing distance
      pool_type: 'average' or 'max', only for multi-query case
    """
    feats, ids, cams, im_names, marks = self.extract_feat(normalize_feat)
    q_inds = marks == 0
    g_inds = marks == 1
    mq_inds = marks == 2

    q_g_dist = compute_dist(feats[q_inds], feats[g_inds], type='euclidean')
    mAP, cmc_scores = self.eval_map_cmc(
      q_g_dist=q_g_dist,
      q_ids=ids[q_inds], g_ids=ids[g_inds],
      q_cams=cams[q_inds], g_cams=cams[g_inds],
      separate_camera_set=self.separate_camera_set,
      single_gallery_shot=self.single_gallery_shot,
      first_match_break=self.first_match_break,
      topk=10)

    # multi-query

    mq_mAP, mq_cmc_scores = None, None
    if any(mq_inds):
      mq_ids = ids[mq_inds]
      mq_cams = cams[mq_inds]
      mq_feats = feats[mq_inds]
      unique_mq_ids_cams = defaultdict(list)
      for ind, (id, cam) in enumerate(zip(mq_ids, mq_cams)):
        unique_mq_ids_cams[(id, cam)].append(ind)
      keys = unique_mq_ids_cams.keys()
      assert pool_type in ['average', 'max']
      pool = np.mean if pool_type == 'average' else np.max
      mq_feats = np.stack([pool(mq_feats[unique_mq_ids_cams[k]], axis=0)
                           for k in keys])
      mq_g_dist = compute_dist(mq_feats, feats[g_inds], type='euclidean')
      mq_mAP, mq_cmc_scores = self.eval_map_cmc(
        q_g_dist=mq_g_dist,
        q_ids=np.array(zip(*keys)[0]), g_ids=ids[g_inds],
        q_cams=np.array(zip(*keys)[1]), g_cams=cams[g_inds],
        separate_camera_set=self.separate_camera_set,
        single_gallery_shot=self.single_gallery_shot,
        first_match_break=self.first_match_break,
        topk=10)

    return mAP, cmc_scores, mq_mAP, mq_cmc_scores
