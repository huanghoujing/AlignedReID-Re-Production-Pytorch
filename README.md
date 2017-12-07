**Not Succeed Yet! Bug May Exist in Computing Local Distance!**

This project is in progress. I try to re-produce the impressive results of paper [AlignedReID: Surpassing Human-Level Performance in Person Re-Identification](https://arxiv.org/abs/1711.08184) using [pytorch](https://github.com/pytorch/pytorch).


# TODO List

- Models
  - [x] ResNet-50
- Loss
  - [x] Triplet Global Loss
  - [ ] Triplet Local Loss
  - [ ] Identification Loss
  - [ ] Mutual Loss
- Testing
  - [ ] Re-Ranking
- Speed
  - [ ] Speed up forward & backward


Current results on Market1501:

|   | Rank-1 (%) | mAP (%) |
| --- | --- | --- |
| Triplet Global Loss| 82.72 | 66.85 |
| Triplet Global + Local Loss| 79.63 | 63.18 |


# Installation

It's recommended that you create and enter a python virtual environment before installing our package.

```bash
git clone https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch.git
```

## Requirements

I use Python 2.7 and Pytorch 0.1.12. For installing Pytorch 0.1.12, follow the [official guide](http://pytorch.org/previous-versions/). Other packages are specified in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Then install this project:

```bash
python setup.py install --record installed_files.txt
```

# Dataset Preparation

Inspired by Tong Xiao's [open-reid](https://github.com/Cysu/open-reid) project, dataset directories are refactored to support a unified dataset interface.

Transformed dataset has following features
- All used images, including training and testing images, are inside the same folder named `images`
- The train/val/test partitions are recorded in a file named `partitions.pkl` which is a dict with the following keys
  - `'trainval_im_names'`
  - `'trainval_ids2labels'`
  - `'train_im_names'`
  - `'train_ids2labels'`
  - `'val_im_names'`
  - `'val_marks'`
  - `'test_im_names'`
  - `'test_marks'`
- Validation set consists of 100 persons (configurable during transforming dataset) unseen in training set, and validation follows the same ranking protocol of testing.
- Each val or test image is accompanied by a mark denoting whether it is from
  - query (mark == 0), or
  - gallery (mark == 1), or
  - multi query (mark == 2) set

## Market1501

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1CaWH7_csm9aDyTVgjs7_3dlZIWqoBlv4) or [BaiduYun](https://pan.baidu.com/s/1nvOhpot). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the Market1501 dataset from [here](http://www.liangzheng.org/Project/project_reid.html). Run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_market1501.py \
--zip_file ~/Dataset/market1501/Market-1501-v15.09.15.zip \
--save_dir ~/Dataset/market1501
```

## CUHK03

We follow the new training/testing protocol proposed in paper
```
@article{zhong2017re,
  title={Re-ranking Person Re-identification with k-reciprocal Encoding},
  author={Zhong, Zhun and Zheng, Liang and Cao, Donglin and Li, Shaozi},
  booktitle={CVPR},
  year={2017}
}
```
Details of the new protocol can be found [here](https://github.com/zhunzhong07/person-re-ranking).

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1Ssp9r4g8UbGveX-9JvHmjpcesvw90xIF) or [BaiduYun](https://pan.baidu.com/s/1hsB0pIc). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the CUHK03 dataset from [here](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html). Then download the training/testing partition file from [Google Drive](https://drive.google.com/open?id=14lEiUlQDdsoroo8XJvQ3nLZDIDeEizlP) or [BaiduYun](https://pan.baidu.com/s/1miuxl3q). This partition file specifies which images are in training, query or gallery set. Finally run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_cuhk03.py \
--zip_file ~/Dataset/cuhk03/cuhk03_release.zip \
--train_test_partition_file ~/Dataset/cuhk03/re_ranking_train_test_split.pkl \
--save_dir ~/Dataset/cuhk03
```


## DukeMTMC-reID

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1P9Jr0en0HBu_cZ7txrb2ZA_dI36wzXbS) or [BaiduYun](https://pan.baidu.com/s/1miIdEek). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the DukeMTMC-reID dataset from [here](https://github.com/layumi/DukeMTMC-reID_evaluation). Run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_duke.py \
--zip_file ~/Dataset/duke/DukeMTMC-reID.zip \
--save_dir ~/Dataset/duke
```

## Configure Dataset Path in Training Script

The training code requires you to configure the dataset paths. In `script/tri_loss/train_cfg.py`, modify the following snippet according to your saving paths used in preparing datasets.

```python
# In file script/tri_loss/train_cfg.py

if self.dataset == 'market1501':
  self.im_dir = osp.expanduser('~/Dataset/market1501/images')
  self.partition_file = osp.expanduser('~/Dataset/market1501/partitions.pkl')
elif self.dataset == 'cuhk03':
  self.im_type = ['detected', 'labeled'][0]
  self.im_dir = osp.expanduser(osp.join('~/Dataset/cuhk03', self.im_type, 'images'))
  self.partition_file = osp.expanduser(osp.join('~/Dataset/cuhk03', self.im_type, 'partitions.pkl'))
elif self.dataset == 'duke':
  self.im_dir = osp.expanduser('~/Dataset/duke/images')
  self.partition_file = osp.expanduser('~/Dataset/duke/partitions.pkl')
```

# Training Examples

**NOTE:** After changing files in directory `aligned_reid`, you have to install the package again by `python setup.py install --record installed_files.txt`. Because scripts that import from this `aligned_reid` package in fact import from site-package, you have to install to update the site-package.

To train and test ResNet-50 + Triplet Global Loss on Market1501:

```bash
python script/tri_loss/train.py \
-d '(0,)' \
-r 1 \
--dataset market1501 \
-glw 1.0 \
-llw 0 \
--log_to_file
```

To train and test ResNet-50 + Triplet Global Loss + Triplet Local Loss on Market1501:

```bash
python script/tri_loss/train.py \
-d '(0,)' \
-r 1 \
--dataset market1501 \
-glw 1.0 \
-llw 1.0 \
--log_to_file
```

You can run the [TensorBoard](https://github.com/lanpa/tensorboard-pytorch) to watch the loss curves etc during training. E.g.

```bash
# Modify the path for `--logdir` accordingly.
tensorboard --logdir exp/tri_loss/market1501/train/g1.0000_l1.0000/run1/tensorboard
```

For more usage of TensorBoard, see the help:

```bash
tensorboard --help
```
