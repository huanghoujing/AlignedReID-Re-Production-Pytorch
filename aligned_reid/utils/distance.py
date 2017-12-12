"""NOTE the input/output shape of methods."""
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
  dist = dist[-1, -1]
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
