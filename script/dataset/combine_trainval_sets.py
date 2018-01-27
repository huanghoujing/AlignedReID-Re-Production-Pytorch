from __future__ import print_function

import sys
sys.path.insert(0, '.')

import os.path as osp

ospeu = osp.expanduser
ospj = osp.join
ospap = osp.abspath

from collections import defaultdict
import shutil

from aligned_reid.utils.utils import may_make_dir
from aligned_reid.utils.utils import save_pickle
from aligned_reid.utils.utils import load_pickle

from aligned_reid.utils.dataset_utils import new_im_name_tmpl
from aligned_reid.utils.dataset_utils import parse_im_name


def move_ims(
    ori_im_paths,
    new_im_dir,
    parse_im_name,
    new_im_name_tmpl,
    new_start_id):
  """Rename and move images to new directory."""
  ids = [parse_im_name(osp.basename(p), 'id') for p in ori_im_paths]
  cams = [parse_im_name(osp.basename(p), 'cam') for p in ori_im_paths]

  unique_ids = list(set(ids))
  unique_ids.sort()
  id_mapping = dict(
    zip(unique_ids, range(new_start_id, new_start_id + len(unique_ids))))

  new_im_names = []
  cnt = defaultdict(int)
  for im_path, id, cam in zip(ori_im_paths, ids, cams):
    new_id = id_mapping[id]
    cnt[(new_id, cam)] += 1
    new_im_name = new_im_name_tmpl.format(new_id, cam, cnt[(new_id, cam)] - 1)
    shutil.copy(im_path, ospj(new_im_dir, new_im_name))
    new_im_names.append(new_im_name)
  return new_im_names, id_mapping


def combine_trainval_sets(
    im_dirs,
    partition_files,
    save_dir):
  new_im_dir = ospj(save_dir, 'trainval_images')
  may_make_dir(new_im_dir)
  new_im_names = []
  new_start_id = 0
  for im_dir, partition_file in zip(im_dirs, partition_files):
    partitions = load_pickle(partition_file)
    im_paths = [ospj(im_dir, n) for n in partitions['trainval_im_names']]
    im_paths.sort()
    new_im_names_, id_mapping = move_ims(
      im_paths, new_im_dir, parse_im_name, new_im_name_tmpl, new_start_id)
    new_start_id += len(id_mapping)
    new_im_names += new_im_names_

  new_ids = range(new_start_id)
  partitions = {'trainval_im_names': new_im_names,
                'trainval_ids2labels': dict(zip(new_ids, new_ids)),
                }
  partition_file = ospj(save_dir, 'partitions.pkl')
  save_pickle(partitions, partition_file)
  print('Partition file saved to {}'.format(partition_file))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(
    description="Combine Trainval Set of Market1501, CUHK03, DukeMTMC-reID")

  # Image directory and partition file of transformed datasets

  parser.add_argument(
    '--market1501_im_dir',
    type=str,
    default=ospeu('~/Dataset/market1501/images')
  )
  parser.add_argument(
    '--market1501_partition_file',
    type=str,
    default=ospeu('~/Dataset/market1501/partitions.pkl')
  )

  cuhk03_im_type = ['detected', 'labeled'][0]
  parser.add_argument(
    '--cuhk03_im_dir',
    type=str,
    # Remember to select the detected or labeled set.
    default=ospeu('~/Dataset/cuhk03/{}/images'.format(cuhk03_im_type))
  )
  parser.add_argument(
    '--cuhk03_partition_file',
    type=str,
    # Remember to select the detected or labeled set.
    default=ospeu('~/Dataset/cuhk03/{}/partitions.pkl'.format(cuhk03_im_type))
  )

  parser.add_argument(
    '--duke_im_dir',
    type=str,
    default=ospeu('~/Dataset/duke/images'))
  parser.add_argument(
    '--duke_partition_file',
    type=str,
    default=ospeu('~/Dataset/duke/partitions.pkl')
  )

  parser.add_argument(
    '--save_dir',
    type=str,
    default=ospeu('~/Dataset/market1501_cuhk03_duke')
  )

  args = parser.parse_args()

  im_dirs = [
    ospap(ospeu(args.market1501_im_dir)),
    ospap(ospeu(args.cuhk03_im_dir)),
    ospap(ospeu(args.duke_im_dir))
  ]
  partition_files = [
    ospap(ospeu(args.market1501_partition_file)),
    ospap(ospeu(args.cuhk03_partition_file)),
    ospap(ospeu(args.duke_partition_file))
  ]

  save_dir = ospap(ospeu(args.save_dir))
  may_make_dir(save_dir)

  combine_trainval_sets(im_dirs, partition_files, save_dir)
