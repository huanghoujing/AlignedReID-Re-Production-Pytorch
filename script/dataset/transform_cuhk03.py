"""Refactor file directories, save/rename images and partition the 
train/val/test set, in order to support the unified dataset interface.
"""

from __future__ import print_function

import sys
sys.path.insert(0, '.')

from zipfile import ZipFile
import os.path as osp
import sys
import h5py
from scipy.misc import imsave
from itertools import chain

from aligned_reid.utils.utils import may_make_dir
from aligned_reid.utils.utils import load_pickle
from aligned_reid.utils.utils import save_pickle

from aligned_reid.utils.dataset_utils import partition_train_val_set
from aligned_reid.utils.dataset_utils import new_im_name_tmpl
from aligned_reid.utils.dataset_utils import parse_im_name


def save_images(mat_file, save_dir, new_im_name_tmpl):
  def deref(mat, ref):
    return mat[ref][:].T

  def dump(mat, refs, pid, cam, im_dir):
    """Save the images of a person under one camera."""
    for i, ref in enumerate(refs):
      im = deref(mat, ref)
      if im.size == 0 or im.ndim < 2: break
      fname = new_im_name_tmpl.format(pid, cam, i)
      imsave(osp.join(im_dir, fname), im)

  mat = h5py.File(mat_file, 'r')
  labeled_im_dir = osp.join(save_dir, 'labeled/images')
  detected_im_dir = osp.join(save_dir, 'detected/images')
  all_im_dir = osp.join(save_dir, 'all/images')

  may_make_dir(labeled_im_dir)
  may_make_dir(detected_im_dir)
  may_make_dir(all_im_dir)

  # loop through camera pairs
  pid = 0
  for labeled, detected in zip(mat['labeled'][0], mat['detected'][0]):
    labeled, detected = deref(mat, labeled), deref(mat, detected)
    assert labeled.shape == detected.shape
    # loop through ids in a camera pair
    for i in range(labeled.shape[0]):
      # We don't care about whether different persons are under same cameras,
      # we only care about the same person being under different cameras or not.
      dump(mat, labeled[i, :5], pid, 0, labeled_im_dir)
      dump(mat, labeled[i, 5:], pid, 1, labeled_im_dir)
      dump(mat, detected[i, :5], pid, 0, detected_im_dir)
      dump(mat, detected[i, 5:], pid, 1, detected_im_dir)
      dump(mat, chain(detected[i, :5], labeled[i, :5]), pid, 0, all_im_dir)
      dump(mat, chain(detected[i, 5:], labeled[i, 5:]), pid, 1, all_im_dir)
      pid += 1
      if pid % 100 == 0:
        sys.stdout.write('\033[F\033[K')
        print('Saving images {}/{}'.format(pid, 1467))


def transform(zip_file, train_test_partition_file, save_dir=None):
  """Save images and partition the train/val/test set.
  """
  print("Extracting zip file")
  root = osp.dirname(osp.abspath(zip_file))
  if save_dir is None:
    save_dir = root
  may_make_dir(save_dir)
  with ZipFile(zip_file) as z:
    z.extractall(path=save_dir)
  print("Extracting zip file done")
  mat_file = osp.join(save_dir, osp.basename(zip_file)[:-4], 'cuhk-03.mat')

  save_images(mat_file, save_dir, new_im_name_tmpl)

  if osp.exists(train_test_partition_file):
    train_test_partition = load_pickle(train_test_partition_file)
  else:
    raise RuntimeError('Train/test partition file should be provided.')

  for im_type in ['detected', 'labeled']:
    trainval_im_names = train_test_partition[im_type]['train_im_names']
    trainval_ids = list(set([parse_im_name(n, 'id')
                             for n in trainval_im_names]))
    # Sort ids, so that id-to-label mapping remains the same when running
    # the code on different machines.
    trainval_ids.sort()
    trainval_ids2labels = dict(zip(trainval_ids, range(len(trainval_ids))))
    train_val_partition = \
      partition_train_val_set(trainval_im_names, parse_im_name, num_val_ids=100)
    train_im_names = train_val_partition['train_im_names']
    train_ids = list(set([parse_im_name(n, 'id')
                          for n in train_val_partition['train_im_names']]))
    # Sort ids, so that id-to-label mapping remains the same when running
    # the code on different machines.
    train_ids.sort()
    train_ids2labels = dict(zip(train_ids, range(len(train_ids))))

    # A mark is used to denote whether the image is from
    #   query (mark == 0), or
    #   gallery (mark == 1), or
    #   multi query (mark == 2) set

    val_marks = [0, ] * len(train_val_partition['val_query_im_names']) \
                + [1, ] * len(train_val_partition['val_gallery_im_names'])
    val_im_names = list(train_val_partition['val_query_im_names']) \
                   + list(train_val_partition['val_gallery_im_names'])
    test_im_names = list(train_test_partition[im_type]['query_im_names']) \
                    + list(train_test_partition[im_type]['gallery_im_names'])
    test_marks = [0, ] * len(train_test_partition[im_type]['query_im_names']) \
                 + [1, ] * len(
      train_test_partition[im_type]['gallery_im_names'])
    partitions = {'trainval_im_names': trainval_im_names,
                  'trainval_ids2labels': trainval_ids2labels,
                  'train_im_names': train_im_names,
                  'train_ids2labels': train_ids2labels,
                  'val_im_names': val_im_names,
                  'val_marks': val_marks,
                  'test_im_names': test_im_names,
                  'test_marks': test_marks}
    partition_file = osp.join(save_dir, im_type, 'partitions.pkl')
    save_pickle(partitions, partition_file)
    print('Partition file for "{}" saved to {}'.format(im_type, partition_file))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="Transform CUHK03 Dataset")
  parser.add_argument(
    '--zip_file',
    type=str,
    default='~/Dataset/cuhk03/cuhk03_release.zip')
  parser.add_argument(
    '--save_dir',
    type=str,
    default='~/Dataset/cuhk03')
  parser.add_argument(
    '--train_test_partition_file',
    type=str,
    default='~/Dataset/cuhk03/re_ranking_train_test_split.pkl')
  args = parser.parse_args()
  zip_file = osp.abspath(osp.expanduser(args.zip_file))
  train_test_partition_file = osp.abspath(osp.expanduser(
    args.train_test_partition_file))
  save_dir = osp.abspath(osp.expanduser(args.save_dir))
  transform(zip_file, train_test_partition_file, save_dir)
