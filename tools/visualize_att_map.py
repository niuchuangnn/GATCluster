import sys
sys.path.insert(0, "./")
import argparse
import os
import torch
from deepcluster.config import Config
from deepcluster.modeling.architectures import build_architecture
from deepcluster.utils.checkpoint import ClusterCheckpointer
from deepcluster.utils.collect_env import collect_env_info
from deepcluster.utils.comm import synchronize, get_rank
from deepcluster.utils.logger import setup_logger
import numpy as np
from deepcluster.utils.visualization import show_examples3
from PIL import Image
import torchvision.transforms.functional as tf
import torchvision.transforms as transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="./configs/stl10/gatcluster.py",
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

    output_dir = cfg.results.output_dir
    checkpointer = ClusterCheckpointer(cfg, model, save_dir=output_dir)

    return_att_map = True

    f = "model_best.pth"
    ckpt = "{}/{}".format(output_dir, f)
    _ = checkpointer.load(ckpt, False)

    data_folder = "datasets/stl10/example_images/"
    save_folder = "./results/stl10/att_maps"

    images = []
    images_color = []
    to_tensor = transforms.ToTensor()
    for i in range(10):
        img_path = "{}/{}.png".format(data_folder, i)
        img_ori = Image.open(img_path)
        images_color.append(np.asarray(img_ori) / 255.0)
        img = tf.to_grayscale(img_ori)
        img = np.array(img).astype(np.float32) / 255.
        img = to_tensor(img)
        images.append(img.unsqueeze(dim=0))

    images_color = np.array(images_color)
    images = torch.cat(images)
    images = images.to(device)
    fea, scores, fea_att, scores_att, att_map = model(images, return_att_map=return_att_map)

    data_show = images_color
    if data_show.shape[1] > 3:
        data_show = data_show.transpose([0, 3, 1, 2])
    scores_np = scores.detach().cpu().numpy()
    att_maps_np = att_map.detach().cpu().numpy()

    show_examples3(data_show, scores_np, att_maps_np, save_folder, save_img=False)


if __name__ == "__main__":
    main()
