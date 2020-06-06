# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""
import sys
sys.path.insert(0, './')
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from deepcluster.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os


import torch
from deepcluster.config import Config
from deepcluster.data import build_data_loader
from deepcluster.solver import make_lr_scheduler, make_optimizer

# from deepcluster.engine.inference import inference
from deepcluster.engine.training import do_train
from deepcluster.modeling.architectures import build_architecture
from deepcluster.utils.checkpoint import ClusterCheckpointer
from deepcluster.utils.collect_env import collect_env_info
from deepcluster.utils.comm import synchronize, get_rank
from deepcluster.utils.imports import import_file
from deepcluster.utils.logger import setup_logger
from deepcluster.utils.miscellaneous import mkdir, save_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(cfg, local_rank, distributed):

    model = build_architecture(cfg.model)
    print(model)
    device = torch.device(cfg.device)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = None

    if "steps" in cfg.solver:
        scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {"iteration": 0}

    output_dir = cfg.results.output_dir

    save_to_disk = get_rank() == 0
    checkpointer = ClusterCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )

    extra_checkpoint_data = checkpointer.load(cfg.model.weight)
    arguments.update(extra_checkpoint_data)

    data_loader = build_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.solver.checkpoint_period

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        cfg,
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="./configs/stl10/gatcluster.py",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg = Config.fromfile(args.config_file)

    num_train = cfg.num_train
    base_output_dir = cfg.results.output_dir
    for n in range(num_train):
        cfg.results.output_dir = "{}_{}".format(base_output_dir, n)
        output_dir = "{}_{}".format(base_output_dir, n)

        if os.path.exists("{}/{}".format(output_dir, "model_final.pth")):
            continue

        if output_dir:
            mkdir(output_dir)

        logger_name = "deep_cluster_{}".format(n)
        cfg.logger_name = logger_name

        logger = setup_logger(logger_name, output_dir, get_rank())
        logger.info("Using {} GPUs".format(num_gpus))
        logger.info(args)

        logger.info("Collecting env info (might take some time)")
        logger.info("\n" + collect_env_info())

        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))

        output_config_path = os.path.join(output_dir, 'config.py')
        logger.info("Saving config into: {}".format(output_config_path))
        # save overloaded model config in the output directory
        save_config(cfg, output_config_path)

        train(cfg, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
