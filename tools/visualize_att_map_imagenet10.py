# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import sys
sys.path.insert(0, "./")
from deepcluster.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from deepcluster.config import Config
from deepcluster.data import build_data_loader

from deepcluster.modeling.architectures import build_architecture
from deepcluster.utils.checkpoint import ClusterCheckpointer
from deepcluster.utils.collect_env import collect_env_info
from deepcluster.utils.comm import synchronize, get_rank
from deepcluster.utils.logger import setup_logger
from deepcluster.utils.miscellaneous import mkdir
from tqdm import tqdm
from deepcluster.utils.evaluation import acc, calculate_nmi, calculate_ari, calculate_acc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from deepcluster.utils.visualization import show_tensor_imgs, show_examples, show_examples3
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="./configs/imagenet10/gatcluster.py",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg = Config.fromfile(args.config_file)

    save_dir = ""
    logger = setup_logger("deep_cluster", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_architecture(cfg.model)
    device = torch.device(cfg.device)
    model.to(device)
    model.eval()

    return_att_map = True

    state_dict = torch.load("./results/imagenet10/gatcluster/model_imagenet10.pth")["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    data_folder = "datasets/imagenet10/example_images/"
    save_folder = "./results/imagenet10/att_maps"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    input_size = 128
    images = []
    images_color = []
    to_tensor = transforms.ToTensor()
    for i in range(10):
        img_path = "{}/{}.png".format(data_folder, i)
        img_ori = Image.open(img_path)
        images_color.append(np.asarray(img_ori).transpose([2, 0, 1]) / 255.)
        img = tf.resize(img_ori, [input_size, input_size])
        img = tf.to_grayscale(img)
        img = np.array(img).astype(np.float32) / 255.
        img = to_tensor(img)
        img = tf.normalize(img, mean=[0.449], std=[0.226])
        images.append(img.unsqueeze(dim=0))

    images = torch.cat(images)
    images = images.to(device)
    fea, scores, fea_att, scores_att, att_map = model(images, return_att_map=return_att_map)

    data_show = images_color

    scores_np = scores.detach().cpu().numpy()
    att_maps_np = att_map.detach().cpu().numpy()

    show_examples3(data_show, scores_np, att_maps_np, save_folder, save_img=False)


if __name__ == "__main__":
    main()
