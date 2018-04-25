"""Mapping original image name (relative image path) -> my new image name.
The mapping is corresponding to transform_duke.py.
"""

from __future__ import print_function

import sys
sys.path.insert(0, '.')

import os.path as osp
from collections import defaultdict

from aligned_reid.utils.utils import save_pickle
from aligned_reid.utils.dataset_utils import get_im_names
from aligned_reid.utils.dataset_utils import new_im_name_tmpl


def parse_original_im_name(img_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    parsed = int(img_name[:4])
  else:
    parsed = int(img_name[6])
  return parsed


def map_im_names(ori_im_names, parse_im_name, new_im_name_tmpl):
  """Map original im names to new im names."""
  cnt = defaultdict(int)
  new_im_names = []
  for im_name in ori_im_names:
    im_name = osp.basename(im_name)
    id = parse_im_name(im_name, 'id')
    cam = parse_im_name(im_name, 'cam')
    cnt[(id, cam)] += 1
    new_im_name = new_im_name_tmpl.format(id, cam, cnt[(id, cam)] - 1)
    new_im_names.append(new_im_name)
  return new_im_names


def save_im_name_mapping(raw_dir, ori_to_new_im_name_file):
  im_names = []
  for dir_name in ['bounding_box_train', 'bounding_box_test', 'query']:
    im_names_ = get_im_names(osp.join(raw_dir, dir_name), return_path=False, return_np=False)
    im_names_.sort()
    # Images in different original directories may have same names,
    # so here we use relative paths as original image names.
    im_names_ = [osp.join(dir_name, n) for n in im_names_]
    im_names += im_names_
  new_im_names = map_im_names(im_names, parse_original_im_name, new_im_name_tmpl)
  ori_to_new_im_name = dict(zip(im_names, new_im_names))
  save_pickle(ori_to_new_im_name, ori_to_new_im_name_file)
  print('File saved to {}'.format(ori_to_new_im_name_file))

  ##################
  # Just Some Info #
  ##################

  print('len(im_names)', len(im_names))
  print('len(set(im_names))', len(set(im_names)))
  print('len(set(new_im_names))', len(set(new_im_names)))
  print('len(ori_to_new_im_name)', len(ori_to_new_im_name))

  bounding_box_train_im_names = get_im_names(osp.join(raw_dir, 'bounding_box_train'), return_path=False, return_np=False)
  bounding_box_test_im_names = get_im_names(osp.join(raw_dir, 'bounding_box_test'), return_path=False, return_np=False)
  query_im_names = get_im_names(osp.join(raw_dir, 'query'), return_path=False, return_np=False)

  print('set(bounding_box_train_im_names).isdisjoint(set(bounding_box_test_im_names))',
        set(bounding_box_train_im_names).isdisjoint(set(bounding_box_test_im_names)))
  print('set(bounding_box_train_im_names).isdisjoint(set(query_im_names))',
        set(bounding_box_train_im_names).isdisjoint(set(query_im_names)))

  print('set(bounding_box_test_im_names).isdisjoint(set(query_im_names))',
        set(bounding_box_test_im_names).isdisjoint(set(query_im_names)))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="Mapping DukeMTMC-reID Image Names")
  parser.add_argument('--raw_dir', type=str, default=osp.expanduser('~/Dataset/duke/DukeMTMC-reID'))
  parser.add_argument('--ori_to_new_im_name_file', type=str, default=osp.expanduser('~/Dataset/duke/ori_to_new_im_name.pkl'))
  args = parser.parse_args()
  save_im_name_mapping(args.raw_dir, args.ori_to_new_im_name_file)