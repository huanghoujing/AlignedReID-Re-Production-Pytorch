from aligned_reid.utils.utils import load_pickle

from .TrainSet import TrainSet
from .TestSet import TestSet


def create_dataset(name='market1501',
                   part='trainval',
                   partition_file=None,
                   **kwargs):
  assert name in ['market1501', 'cuhk03', 'duke']
  assert part in ['trainval', 'train', 'val', 'test']
  cmc_kwargs = dict(separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
  partitions = load_pickle(partition_file)
  if part == 'trainval':
    return TrainSet(
      im_names=partitions['trainval_im_names'],
      ids2labels=partitions['trainval_ids2labels'],
      **kwargs)
  if part == 'train':
    return TrainSet(
      im_names=partitions['train_im_names'],
      ids2labels=partitions['train_ids2labels'],
      **kwargs)
  if part == 'val':
    kwargs.update(cmc_kwargs)
    return TestSet(
      im_names=partitions['val_im_names'],
      marks=partitions['val_marks'],
      **kwargs)
  if part == 'test':
    kwargs.update(cmc_kwargs)
    return TestSet(
      im_names=partitions['test_im_names'],
      marks=partitions['test_marks'],
      **kwargs)