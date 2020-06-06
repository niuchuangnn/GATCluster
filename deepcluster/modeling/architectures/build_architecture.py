from .gaussian_attention_cluster import GaussianAttentionCluster


def build_architecture(arc_cfg_ori):
    arc_cfg = arc_cfg_ori.copy()
    arc_type = arc_cfg.pop("type")
    if arc_type == "gattcluster":
        return GaussianAttentionCluster(**arc_cfg)
    else:
        raise TypeError
