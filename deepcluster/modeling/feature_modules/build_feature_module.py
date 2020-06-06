from .convnet import ConvNet
from .mlp import MLP
from .resnet import ResNet5gTrunk
from .resnet_all import resnet34


def build_feature_module(fea_cfg_ori):
    fea_cfg = fea_cfg_ori.copy()
    fea_type = fea_cfg.pop("type")
    if fea_type == "mlp":
        return MLP(**fea_cfg)
    elif fea_type == "convnet":
        return ConvNet(**fea_cfg)
    elif fea_type == "resnet5g":
        return ResNet5gTrunk(**fea_cfg)
    elif fea_type == "resnet34":
        return resnet34()
    else:
        raise TypeError