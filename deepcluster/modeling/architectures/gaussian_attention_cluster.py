import torch.nn as nn
from ..feature_modules import build_feature_module
from ..gaussian_att_cluster_head import build_gaussian_att_cluster_head


class GaussianAttentionCluster(nn.Module):

    def __init__(self, feature, gaussian_att_cluster_head, **kwargs):
        super(GaussianAttentionCluster, self).__init__()
        self.feature_module = build_feature_module(feature)
        self.gaussian_att_cluster_head = build_gaussian_att_cluster_head(gaussian_att_cluster_head)

    def forward_train(self, fea, target=None):
        loss = self.gaussian_att_cluster_head.loss(fea, target)

        return loss

    def forward_test(self, fea, return_target=False, return_att_map=False):
        out = self.gaussian_att_cluster_head(fea, return_target, return_att_map)
        return out

    def forward(self, input_data, target=None, return_target=False, return_att_map=False, **kwargs):
        fea = self.feature_module(input_data)
        if self.training:
            return self.forward_train(fea, target)
        else:
            return self.forward_test(fea, return_target, return_att_map)
