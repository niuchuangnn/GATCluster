from .gaussian_att_cluster_head import GaussianAttentionClusterHead


def build_gaussian_att_cluster_head(cfg):

    return GaussianAttentionClusterHead(**cfg)