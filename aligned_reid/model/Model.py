import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .resnet import resnet50


class Model(nn.Module):
  def __init__(self, local_conv_out_channels=128, num_classes=None):
    super(Model, self).__init__()
    self.base = resnet50(pretrained=True)
    planes = 2048
    self.local_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
    self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
    self.local_relu = nn.ReLU(inplace=True)

    if num_classes is not None:
      self.fc = nn.Linear(planes, num_classes)
      init.normal(self.fc.weight, std=0.001)
      init.constant(self.fc.bias, 0)

  def forward(self, x):
    """
    Returns:
      global_feat: shape [N, C]
      local_feat: shape [N, H, c]
    """
    # shape [N, C, H, W]
    feat = self.base(x)
    global_feat = F.avg_pool2d(feat, feat.size()[2:])
    # shape [N, C]
    global_feat = global_feat.view(global_feat.size(0), -1)
    # shape [N, C, H, 1]
    local_feat = torch.mean(feat, -1, keepdim=True)
    local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
    # shape [N, H, c]
    local_feat = local_feat.squeeze(-1).permute(0, 2, 1)

    if hasattr(self, 'fc'):
      logits = self.fc(global_feat)
      return global_feat, local_feat, logits

    return global_feat, local_feat
