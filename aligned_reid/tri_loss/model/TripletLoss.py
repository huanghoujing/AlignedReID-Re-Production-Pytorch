from torch import nn
from torch.autograd import Variable


class TripletLoss(object):
  """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid). 
  Related Triplet Loss theory can be found in paper 'In Defense of the Triplet 
  Loss for Person Re-Identification'."""
  def __init__(self, margin=None):
    self.margin = margin
    if margin is not None:
      self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    else:
      self.ranking_loss = nn.SoftMarginLoss()

  def __call__(self, dist_ap, dist_an):
    """
    Args:
      dist_ap: pytorch Variable, distance between anchor and positive sample, 
        shape [N]
      dist_an: pytorch Variable, distance between anchor and negative sample, 
        shape [N]
    Returns:
      loss: pytorch Variable, with shape [1]
      prec: a scalar. precision, the percentage of 
        `dist(anchor, neg) > dist(anchor, pos)`
      ret: a scalar. If `self.margin is None`, average of 
        `dist(anchor, neg) - dist(anchor, pos)`; Otherwise, percentage of 
        `dist(anchor, neg) > dist(anchor, pos) + margin`.
    """
    y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
    if self.margin is not None:
      loss = self.ranking_loss(dist_an, dist_ap, y)
    else:
      loss = self.ranking_loss(dist_an - dist_ap, y)
    prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
    if self.margin is not None:
      # Percentage of satisfying the margin
      ret = (dist_an.data > dist_ap.data + self.margin).sum() * 1. / y.size(0)
    else:
      # Average distance difference
      ret = float((dist_an.data - dist_ap.data).sum()) / y.size(0)
    return loss, prec, ret
