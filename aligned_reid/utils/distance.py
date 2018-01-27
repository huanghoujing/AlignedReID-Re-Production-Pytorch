"""Numpy version of euclidean distance, shortest distance, etc.
Notice the input/output shape of methods, so that you can better understand
the meaning of these methods."""
import numpy as np


def normalize(nparray, order=2, axis=0):
  """Normalize a N-D numpy array along the specified axis."""
  norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
  return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):
  """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
  assert type in ['cosine', 'euclidean']
  if type == 'cosine':
    array1 = normalize(array1, axis=1)
    array2 = normalize(array2, axis=1)
    dist = np.matmul(array1, array2.T)
    return dist
  else:
    # shape [m1, 1]
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    dist = np.sqrt(squared_dist)
    return dist


def shortest_dist(dist_mat):
  """Parallel version.
  Args:
    dist_mat: numpy array, available shape
      1) [m, n]
      2) [m, n, N], N is batch size
      3) [m, n, *], * can be arbitrary additional dimensions
  Returns:
    dist: three cases corresponding to `dist_mat`
      1) scalar
      2) numpy array, with shape [N]
      3) numpy array with shape [*]
  """
  m, n = dist_mat.shape[:2]
  dist = np.zeros_like(dist_mat)
  for i in range(m):
    for j in range(n):
      if (i == 0) and (j == 0):
        dist[i, j] = dist_mat[i, j]
      elif (i == 0) and (j > 0):
        dist[i, j] = dist[i, j - 1] + dist_mat[i, j]
      elif (i > 0) and (j == 0):
        dist[i, j] = dist[i - 1, j] + dist_mat[i, j]
      else:
        dist[i, j] = \
          np.min(np.stack([dist[i - 1, j], dist[i, j - 1]], axis=0), axis=0) \
          + dist_mat[i, j]
  # I ran into memory disaster when returning this reference! I still don't
  # know why.
  # dist = dist[-1, -1]
  dist = dist[-1, -1].copy()
  return dist


def meta_local_dist(x, y):
  """
  Args:
    x: numpy array, with shape [m, d]
    y: numpy array, with shape [n, d]
  Returns:
    dist: scalar
  """
  eu_dist = compute_dist(x, y, 'euclidean')
  dist_mat = (np.exp(eu_dist) - 1.) / (np.exp(eu_dist) + 1.)
  dist = shortest_dist(dist_mat[np.newaxis])[0]
  return dist


# Tooooooo slow!
def serial_local_dist(x, y):
  """
  Args:
    x: numpy array, with shape [M, m, d]
    y: numpy array, with shape [N, n, d]
  Returns:
    dist: numpy array, with shape [M, N]
  """
  M, N = x.shape[0], y.shape[0]
  dist_mat = np.zeros([M, N])
  for i in range(M):
    for j in range(N):
      dist_mat[i, j] = meta_local_dist(x[i], y[j])
  return dist_mat


def parallel_local_dist(x, y):
  """Parallel version.
  Args:
    x: numpy array, with shape [M, m, d]
    y: numpy array, with shape [N, n, d]
  Returns:
    dist: numpy array, with shape [M, N]
  """
  M, m, d = x.shape
  N, n, d = y.shape
  x = x.reshape([M * m, d])
  y = y.reshape([N * n, d])
  # shape [M * m, N * n]
  dist_mat = compute_dist(x, y, type='euclidean')
  dist_mat = (np.exp(dist_mat) - 1.) / (np.exp(dist_mat) + 1.)
  # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
  dist_mat = dist_mat.reshape([M, m, N, n]).transpose([1, 3, 0, 2])
  # shape [M, N]
  dist_mat = shortest_dist(dist_mat)
  return dist_mat


def local_dist(x, y):
  if (x.ndim == 2) and (y.ndim == 2):
    return meta_local_dist(x, y)
  elif (x.ndim == 3) and (y.ndim == 3):
    return parallel_local_dist(x, y)
  else:
    raise NotImplementedError('Input shape not supported.')


def low_memory_matrix_op(
    func,
    x, y,
    x_split_axis, y_split_axis,
    x_num_splits, y_num_splits,
    verbose=False):
  """
  For matrix operation like multiplication, in order not to flood the memory 
  with huge data, split matrices into smaller parts (Divide and Conquer). 
  
  Note: 
    If still out of memory, increase `*_num_splits`.
  
  Args:
    func: a matrix function func(x, y) -> z with shape [M, N]
    x: numpy array, the dimension to split has length M
    y: numpy array, the dimension to split has length N
    x_split_axis: The axis to split x into parts
    y_split_axis: The axis to split y into parts
    x_num_splits: number of splits. 1 <= x_num_splits <= M
    y_num_splits: number of splits. 1 <= y_num_splits <= N
    verbose: whether to print the progress
    
  Returns:
    mat: numpy array, shape [M, N]
  """

  if verbose:
    import sys
    import time
    printed = False
    st = time.time()
    last_time = time.time()

  mat = [[] for _ in range(x_num_splits)]
  for i, part_x in enumerate(
      np.array_split(x, x_num_splits, axis=x_split_axis)):
    for j, part_y in enumerate(
        np.array_split(y, y_num_splits, axis=y_split_axis)):
      part_mat = func(part_x, part_y)
      mat[i].append(part_mat)

      if verbose:
        if not printed:
          printed = True
        else:
          # Clean the current line
          sys.stdout.write("\033[F\033[K")
        print('Matrix part ({}, {}) / ({}, {}), +{:.2f}s, total {:.2f}s'
              .format(i + 1, j + 1, x_num_splits, y_num_splits,
                      time.time() - last_time, time.time() - st))
        last_time = time.time()
    mat[i] = np.concatenate(mat[i], axis=1)
  mat = np.concatenate(mat, axis=0)
  return mat