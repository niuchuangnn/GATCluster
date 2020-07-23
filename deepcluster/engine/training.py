# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from deepcluster.utils.comm import get_world_size
from deepcluster.utils.metric_logger import MetricLogger
from deepcluster.data import build_data_loader, build_dataset
import random
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
from deepcluster.utils.evaluation import acc, calculate_nmi, calculate_ari
from deepcluster.utils.visualization import show_tensor_imgs
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    cfg,
):
    logger = logging.getLogger("{}.trainer".format(cfg.logger_name))
    logger.info("Start training")
    target_sub_batch_size = cfg.solver.target_sub_batch_size
    train_batch_size = cfg.solver.train_batch_size
    train_sub_batch_size = cfg.solver.train_sub_batch_size

    sim_loss = cfg.solver.sim_loss
    ent_loss = cfg.solver.ent_loss
    rel_loss = cfg.solver.rel_loss
    att_loss = cfg.solver.att_loss

    data_loader_test = build_data_loader(
        cfg,
        is_train=False,
        is_distributed=False,
        start_iter=0,
    )

    num_repeat = cfg.solver.num_repeat
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    best_acc = 0
    best_iter = None

    for iteration, (images_ori, images_trans, _, idx) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        if scheduler is not None:
            scheduler.step()

        # Generate ground truth.
        model.eval()

        num_imgs = images_ori.shape[0]
        target_iters = int(math.ceil(float(num_imgs) / target_sub_batch_size))

        feas = []
        probs = []
        for ti in range(target_iters):
            start_idx = ti*target_sub_batch_size
            end_idx = min((ti+1)*target_sub_batch_size, num_imgs)
            images_ori_batch = images_ori[start_idx:end_idx].to(device)
            gt_fea, gt_prob = model(images_ori_batch, return_target=True)
            feas.append(gt_fea)
            probs.append(gt_prob)

        if len(feas) == 1:
            feas = feas[0]
        else:
            feas = torch.cat(feas, dim=0)

        if len(probs) == 1:
            probs = probs[0]
        else:
            probs = torch.cat(probs, dim=0)

        if sim_loss:
            gt_feas_ori = feas
            gt_probs_ori = model.gaussian_att_cluster_head.compute_balance_socres(probs)
        else:
            gt_feas_ori = None
            gt_probs_ori = None

        if rel_loss:
            model.gaussian_att_cluster_head.compute_kmeans(probs)
            gt_rels = model.gaussian_att_cluster_head.compute_relation_target(probs)
        else:
            gt_rels = None

        if att_loss:
            gt_feas_att = feas
            gt_probs_att = model.gaussian_att_cluster_head.compute_attention_target(probs)
        else:
            gt_feas_att = None

        # Train with the generated ground truth
        model.train()
        img_idx = list(range(num_imgs))
        # Select a set of images for training.
        if num_imgs > train_batch_size:
            num_train = train_batch_size
        else:
            num_train = num_imgs

        # train_sub_iters = int(torch.ceil(float(num_train) / train_sub_batch_size))
        train_sub_iters = num_train // train_sub_batch_size
        if isinstance(images_trans, list):
            num_trans = len(images_trans)
        else:
            num_trans = 1

        for n in range(num_repeat):
            random.shuffle(img_idx)

            for i in range(train_sub_iters):
                start_idx = i*train_sub_batch_size
                end_idx = min((i+1)*train_sub_batch_size, num_train)
                img_idx_i = img_idx[start_idx:end_idx]

                if gt_feas_ori is not None:
                    target_feas_ori = gt_feas_ori[img_idx_i, :]
                else:
                    target_feas_ori = None

                if gt_probs_ori is not None:
                    target_probs_ori = gt_probs_ori[img_idx_i, :]
                else:
                    target_probs_ori = None

                if gt_rels is not None:
                    target_rels = gt_rels[img_idx_i, :][:, img_idx_i]
                else:
                    target_rels = None

                if gt_feas_att is not None:
                    target_feas_att = gt_feas_att[img_idx_i, :]
                else:
                    target_feas_att = None

                if gt_probs_att is not None:
                    target_probs_att = gt_probs_att[img_idx_i, :]
                else:
                    target_probs_att = None

                for it in range(num_trans):
                    if num_trans > 1:
                        images_trans_it = images_trans[it]
                    else:
                        images_trans_it = images_trans

                    imgs_i = images_trans_it[img_idx_i, :, :, :].to(device)

                    loss_dict, model_info = model(imgs_i, target=[target_feas_ori, target_probs_ori, target_rels, target_feas_att, target_probs_att, ent_loss])

                    losses = sum(loss for loss in loss_dict.values())

                    # reduce losses over all GPUs for logging purposes
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters.update(loss=losses_reduced, **loss_dict_reduced)

                    optimizer.zero_grad()
                    losses.backward()

                    # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
                    optimizer.step()

                    batch_time = time.time() - end
                    end = time.time()
                    meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * ((max_iter - iteration) * num_repeat * train_sub_iters * num_trans)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 1 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{model_info}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    model_info=model_info,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        # Test.
        if iteration % checkpoint_period == 0:
            model.eval()

            pred_labels = []
            gt_labels = []
            pred_scores = []

            for _, (images, _, labels, idx) in enumerate(tqdm(data_loader_test)):
                images = images.to(device)
                scores = model(images)

                if isinstance(scores, tuple) or isinstance(scores, list):
                    assert len(scores) == 4
                    fe_ori, scores_ori, fea_att, scores_att = scores

                pred_idx = scores_ori.argmax(dim=1)
                gt_labels.append(labels)
                pred_labels.append(pred_idx)
                pred_scores.append(scores_ori.detach())

            gt_labels = torch.cat(gt_labels).long().cpu().numpy()
            pred_labels = torch.cat(pred_labels).long().cpu().numpy()

            try:
                accuracy = acc(pred_labels, gt_labels)
            except:
                accuracy = -1

            nmi = calculate_nmi(pred_labels, gt_labels)

            ari = calculate_ari(pred_labels, gt_labels)

            if accuracy > best_acc:
                best_acc = accuracy
                best_iter = iteration
                checkpointer.save("model_best", **arguments)

            logger.info("iter: {}, acc: {}, nmi: {}, ari: {}".format(iteration, accuracy, nmi, ari))
            logger.info("Best ACC: {}, iter: {}".format(best_acc, best_iter))

            # Select training samples.
            pred_scores = torch.cat(pred_scores)

            val_sort, idx_sort = torch.sort(pred_scores, dim=0, descending=True)

            if False:
                idx_show = idx_sort[0:10, :].flatten().cpu().numpy()
                data_show = data_loader_test.dataset.data[idx_show, :, :, :] / 255.
                if data_show.shape[1] > 3:
                    data_show = data_show.transpose([0, 3, 1, 2])
                data_show_tensor = torch.from_numpy(data_show)
                im_show = show_tensor_imgs(data_show_tensor, 10, show=False)
                # plt.show()
                im_show_pil = Image.fromarray(im_show)
                im_show_pil.save("{}/{}.png".format(cfg.results.output_dir, iteration))

            model.train()
            checkpointer.save("model_last", **arguments)
            # checkpointer.save("model_{}".format(iteration), **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    logger.info("Best Accuracy: {}, iter: {}".format(best_acc, best_iter))