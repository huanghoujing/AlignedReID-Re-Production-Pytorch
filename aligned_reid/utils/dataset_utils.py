from __future__ import print_function
import os.path as osp
import numpy as np
import glob
from collections import defaultdict
import shutil

new_im_name_tmpl = '{:08d}_{:04d}_{:08d}.jpg'

def parse_im_name(im_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    parsed = int(im_name[:8])
  else:
    parsed = int(im_name[9:13])
  return parsed


def get_im_names(im_dir, pattern='*.jpg', return_np=True, return_path=False):
  """Get the image names in a dir. Optional to return numpy array, paths."""
  im_paths = glob.glob(osp.join(im_dir, pattern))
  im_names = [osp.basename(path) for path in im_paths]
  ret = im_paths if return_path else im_names
  if return_np:
    ret = np.array(ret)
  return ret


def move_ims(ori_im_paths, new_im_dir, parse_im_name, new_im_name_tmpl):
  """Rename and move images to new directory."""
  cnt = defaultdict(int)
  new_im_names = []
  for im_path in ori_im_paths:
    im_name = osp.basename(im_path)
    id = parse_im_name(im_name, 'id')
    cam = parse_im_name(im_name, 'cam')
    cnt[(id, cam)] += 1
    new_im_name = new_im_name_tmpl.format(id, cam, cnt[(id, cam)] - 1)
    shutil.copy(im_path, osp.join(new_im_dir, new_im_name))
    new_im_names.append(new_im_name)
  return new_im_names


def partition_train_val_set(im_names, parse_im_name,
                            num_val_ids=None, val_prop=None, seed=1):
  """Partition the trainval set into train and val set. 
  Args:
    im_names: trainval image names
    parse_im_name: a function to parse id and camera from image name
    num_val_ids: number of ids for val set. If not set, val_prob is used.
    val_prop: the proportion of validation ids
    seed: the random seed to reproduce the partition results. If not to use, 
      then set to `None`.
  Returns:
    a dict with keys (`train_im_names`, 
                      `val_query_im_names`, 
                      `val_gallery_im_names`)
  """
  np.random.seed(seed)
  # Transform to numpy array for slicing.
  if not isinstance(im_names, np.ndarray):
    im_names = np.array(im_names)
  np.random.shuffle(im_names)
  ids = np.array([parse_im_name(n, 'id') for n in im_names])
  cams = np.array([parse_im_name(n, 'cam') for n in im_names])
  unique_ids = np.unique(ids)
  np.random.shuffle(unique_ids)

  # Query indices and gallery indices
  query_inds = []
  gallery_inds = []

  if num_val_ids is None:
    assert 0 < val_prop < 1
    num_val_ids = int(len(unique_ids) * val_prop)
  num_selected_ids = 0
  for unique_id in unique_ids:
    query_inds_ = []
    # The indices of this id in trainval set.
    inds = np.argwhere(unique_id == ids).flatten()
    # The cams that this id has.
    unique_cams = np.unique(cams[inds])
    # For each cam, select one image for query set.
    for unique_cam in unique_cams:
      query_inds_.append(
        inds[np.argwhere(cams[inds] == unique_cam).flatten()[0]])
    gallery_inds_ = list(set(inds) - set(query_inds_))
    # For each query image, if there is no same-id different-cam images in
    # gallery, put it in gallery.
    for query_ind in query_inds_:
      if len(gallery_inds_) == 0 \
          or len(np.argwhere(cams[gallery_inds_] != cams[query_ind])
                     .flatten()) == 0:
        query_inds_.remove(query_ind)
        gallery_inds_.append(query_ind)
    # If no query image is left, leave this id in train set.
    if len(query_inds_) == 0:
      continue
    query_inds.append(query_inds_)
    gallery_inds.append(gallery_inds_)
    num_selected_ids += 1
    if num_selected_ids >= num_val_ids:
      break

  query_inds = np.hstack(query_inds)
  gallery_inds = np.hstack(gallery_inds)
  val_inds = np.hstack([query_inds, gallery_inds])
  trainval_inds = np.arange(len(im_names))
  train_inds = np.setdiff1d(trainval_inds, val_inds)

  train_inds = np.sort(train_inds)
  query_inds = np.sort(query_inds)
  gallery_inds = np.sort(gallery_inds)

  partitions = dict(train_im_names=im_names[train_inds],
                    val_query_im_names=im_names[query_inds],
                    val_gallery_im_names=im_names[gallery_inds])

  return partitions
