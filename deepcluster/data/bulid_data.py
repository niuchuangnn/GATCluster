from deepcluster.data.datasets.stl10 import STL10
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from torch.utils.data import DataLoader
from . import samplers
import torch
import copy
import bisect
import logging
from deepcluster.utils.comm import get_world_size


def build_dataset(data_cfg):
    type = data_cfg.type

    dataset = None

    if type == "stl10":

        normalize = transforms.Normalize(mean=[0.447, 0.440, 0.407], std=[1, 1, 1])
        to_tensor = transforms.ToTensor()
        if data_cfg.train:
            flip = transforms.RandomHorizontalFlip(0.5)
            affine = transforms.RandomAffine(degrees=10, translate=[0.1, 0.1], scale=[0.8, 1.2], shear=10)
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.2)

            T = transforms.Compose([flip, color_jitter, affine, to_tensor, normalize])
        else:
            T = None

        if "num_trans_aug" in data_cfg:
            num_trans_aug = data_cfg.num_trans_aug
        else:
            num_trans_aug = 1

        dataset = STL10(root=data_cfg.root_folder,
                        split=data_cfg.split,
                        show=data_cfg.show,
                        transform=transforms.Compose([to_tensor, normalize]),
                        transform_aug=T,
                        num_trans_aug=num_trans_aug,
                        download=False)
    elif type == "stl10_gray":
        to_gray = tf.to_grayscale
        to_tensor = transforms.ToTensor()
        if data_cfg.train:
            flip = transforms.RandomHorizontalFlip(0.5)
            affine = transforms.RandomAffine(degrees=10, translate=[0.1, 0.1], scale=[0.8, 1.2], shear=10)
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.2)

            T = transforms.Compose([flip, color_jitter, affine, to_gray, to_tensor])
        else:
            T = None

        if "num_trans_aug" in data_cfg:
            num_trans_aug = data_cfg.num_trans_aug
        else:
            num_trans_aug = 1

        dataset = STL10(root=data_cfg.root_folder,
                        split=data_cfg.split,
                        show=data_cfg.show,
                        transform=transforms.Compose([to_gray, to_tensor]),
                        transform_aug=T,
                        num_trans_aug=num_trans_aug,
                        download=False)
    else:
        assert TypeError

    return dataset


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def build_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0):

    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.data_train.ims_per_batch
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = cfg.data_train.shuffle
        num_iters = cfg.solver.max_iter
        dataset = build_dataset(cfg.data_train)
    else:
        images_per_batch = cfg.data_test.ims_per_batch
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0
        dataset = build_dataset(cfg.data_test)

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.data_train.aspect_ratio_grouping else []

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
    )

    num_workers = cfg.num_workers
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
    )

    return data_loader
