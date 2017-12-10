from __future__ import print_function
import torch

import numpy as np

from ...utils.utils import to_scalar
from ...utils.utils import print_array


def normalize(x):
  """
  Args:
    x: pytorch Variable, with shape [*, d]
  Returns:
    x: pytorch Variable, with shape [*, d], normalized to unit length along 
      the last dimension
  """
  x = x / (torch.sqrt(torch.sum(torch.pow(x, 2), -1)).expand_as(x) + 1e-12)
  return x


def euclidean_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
  Returns:
    dist: pytorch Variable, with shape [m, n]
  """
  m, n = x.size(0), y.size(0)
  xx = torch.pow(x, 2).sum(1).expand(m, n)
  yy = torch.pow(y, 2).sum(1).expand(n, m).t()
  dist = xx + yy
  dist.addmm_(1, -2, x, y.t())
  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
  return dist


def shortest_dist(dist_mat):
  """
  Args:
    dist_mat: pytorch Variable, with shape [m, n]
  Returns:
    dist: pytorch Variable, with shape [1]. NOTE: it's normalized by (m + n).
  """
  m, n = dist_mat.size()
  dist = [[0 for _ in range(n)] for _ in range(m)]
  for i in range(m):
    for j in range(n):
      if (i == 0) and (j == 0):
        dist[i][j] = dist_mat[i, j]
      elif (i == 0) and (j > 0):
        dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
      elif (i > 0) and (j == 0):
        dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
      else:
        dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
  dist = dist[-1][-1]
  return dist


def local_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
  Returns:
    dist: pytorch Variable, with shape [1]
  """
  eu_dist = euclidean_dist(x, y)
  # print('eu_dist:\n')
  # print_array(eu_dist.data.cpu().numpy().flatten())
  # Try to skip the exp normalization
  # dist_mat = eu_dist
  dist_mat = (torch.exp(eu_dist) - 1.) / (torch.exp(eu_dist) + 1.)
  # print('dist_mat:\n')
  # print_array(dist_mat.data.cpu().numpy().flatten())
  dist = shortest_dist(dist_mat)
  return dist


def global_loss(tri_loss, global_feat, labels):
  """
  Args:
    tri_loss: a `TripletLoss` object
    global_feat: pytorch Variable, shape [N, C]
    labels: numpy array with shape [N]
  Returns:
    loss: pytorch Variable, with shape [1]
    prec: a scalar. precision, the percentage of 
      `dist(anchor, neg) > dist(anchor, pos)`
    ret: a scalar. either percentage of 
      `dist(anchor, neg) > dist(anchor, pos) + margin`, 
      or average of `dist(anchor, neg) - dist(anchor, pos)`. Determined by 
      whether the `TripletLoss` object uses soft margin or not
    p_inds: a list of scalars; indices of selected hard positive samples; 
      0 <= p_inds[i] <= N - 1
    n_inds: a list of scalars; indices of selected hard negative samples; 
      0 <= n_inds[i] <= N - 1
  """
  global_feat = normalize(global_feat)
  dist_mat = euclidean_dist(global_feat, global_feat)
  N = dist_mat.size(0)
  # dist(anchor, positive), dist(anchor, negative)
  dist_ap, dist_an = [], []
  p_inds, n_inds = [], []
  # For each anchor, find the hardest positive and negative.
  for i in range(N):
    # indices of positive samples
    inds = list(np.arange(N)[labels[i] == labels])
    inds.remove(i)
    # Specifying the `dim` parameter of `torch.max` so that indices are returned
    dist, ind = torch.max(torch.cat([dist_mat[i, ind] for ind in inds]), 0)
    p_inds.append(inds[to_scalar(ind)])
    dist_ap.append(dist)

    # indices of negative samples
    inds = list(np.arange(N)[labels[i] != labels])
    # Specifying the `dim` parameter of `torch.min` so that indices are returned
    dist, ind = torch.min(torch.cat([dist_mat[i, ind] for ind in inds]), 0)
    n_inds.append(inds[to_scalar(ind)])
    dist_an.append(dist)
  dist_ap = torch.cat(dist_ap)
  dist_an = torch.cat(dist_an)

  # -------------------------------------------------------------------------
  # Debug

  # global_dist_ap = dist_ap.data.cpu().numpy().flatten()
  # print('global_dist_ap {:.4f}:\n'.format(np.mean(global_dist_ap)))
  # print_array(global_dist_ap, fmt='{:.4f}', end=' ')
  #
  # global_dist_an = dist_an.data.cpu().numpy().flatten()
  # print('global_dist_an {:.4f}:\n'.format(np.mean(global_dist_an)))
  # print_array(global_dist_an, fmt='{:.4f}', end=' ')

  global_dist_ap = dist_ap.data.cpu().numpy().flatten()
  global_dist_an = dist_an.data.cpu().numpy().flatten()
  end = '\n'
  # end = ', ' # Local distance will be appended to the right
  print('global_dist_ap, global_dist_an: {:.4f}, {:.4f}'.format(
    np.mean(global_dist_ap), np.mean(global_dist_an)), end=end)

  # -------------------------------------------------------------------------

  loss, prec, ret = tri_loss(dist_ap, dist_an)
  return loss, prec, ret, p_inds, n_inds


def local_loss(tri_loss, local_feat, p_inds, n_inds, labels):
  """
  Args:
    tri_loss: a `TripletLoss` object
    local_feat: pytorch Variable, shape [N, H, c] (NOTE THE SHAPE!)
    p_inds: a list of scalars; indices of selected hard positive samples; 
      0 <= p_inds[i] <= N - 1
    n_inds: a list of scalars; indices of selected hard negative samples; 
      0 <= n_inds[i] <= N - 1
    labels: numpy array with shape [N]
  Returns:
    loss: pytorch Variable,with shape [1]
    prec: a scalar. precision, the percentage of 
      `dist(anchor, neg) > dist(anchor, pos)`
    ret: a scalar. either percentage of 
      `dist(anchor, neg) > dist(anchor, pos) + margin`, 
      or average of `dist(anchor, neg) - dist(anchor, pos)`. Determined by 
      whether the `TripletLoss` object uses soft margin or not
  """
  local_feat = normalize(local_feat)
  # dist(anchor, positive), dist(anchor, negative)
  dist_ap, dist_an = [], []
  for lf, p_ind, n_ind in zip(local_feat, p_inds, n_inds):
    dist_ap.append(local_dist(lf, local_feat[p_ind]))
    dist_an.append(local_dist(lf, local_feat[n_ind]))
  dist_ap = torch.cat(dist_ap)
  dist_an = torch.cat(dist_an)

  # -------------------------------------------------------------------------
  # Debug

  # local_dist_ap = dist_ap.data.cpu().numpy().flatten()
  # print('local_dist_ap {:.4f}:\n'.format(np.mean(local_dist_ap)))
  # print_array(local_dist_ap, fmt='{:.4f}', end=' ')

  # local_dist_an = dist_an.data.cpu().numpy().flatten()
  # print('local_dist_an {:.4f}:\n'.format(np.mean(local_dist_an)))
  # print_array(local_dist_an, fmt='{:.4f}', end=' ')

  local_dist_ap = dist_ap.data.cpu().numpy().flatten()
  local_dist_an = dist_an.data.cpu().numpy().flatten()
  print('local_dist_ap, local_dist_an: {:.4f}, {:.4f}'.format(
    np.mean(local_dist_ap), np.mean(local_dist_an)))

  # -------------------------------------------------------------------------

  loss, prec, ret = tri_loss(dist_ap, dist_an)
  return loss, prec, ret
