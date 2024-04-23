#inference
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import torch
from tqdm import tqdm
from fcos_core.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from fcos_core.engine.trainer import foward_detector
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl

from time import time

from .Tsne import tsne
import seaborn as sns
import pandas as pd

#  tsne_torch.py
#
# Implementation of t-SNE in pytorch. The implementation was tested on pytorch
# > 1.0, and it requires Numpy to read files. In order to plot the results,
# a working installation of matplotlib is required.
#
#
# The example can be run by executing: `python tsne_torch.py`
#
#
#  Created by Xiao Li on 23-03-2020.
#  Copyright (c) 2020. All rights reserved.
import numpy as np
import matplotlib.pyplot as pyplot
import argparse
import torch


def compute_on_dataset(cfg, model, data_loader, device, timer=None):
    # model.eval
    for k in model:
        model[k].eval()

    results_dict = {}
    cpu_device = torch.device("cpu")
    feature_list = []
    gt_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            output, feat, gt = foward_detector(cfg, model, images, targets=targets)  # , feat, gt
            if feat != None:
                feature_list.append(feat)
                gt_list.append(gt)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    print(len(feature_list), len(gt_list))
    X = torch.cat(feature_list)
    data = []
    labels = []
    labels_torch = torch.cat(gt_list)
    labes_unique = labels_torch.unique()
    #store all lable != -1
    for lbl in labes_unique:
        idx = labels_torch == lbl
        data.append(X[idx][:580, :])
        labels.append(labels_torch[idx][:580])
    #store top 1000 feats
    # for i in labels:
    #     print(i.shape)
    data = torch.cat(data, dim=0)
    # data_min = torch.min(data, 0)[0]
    # data_max = torch.max(data, 0)[0]
    # print(data_min.shape)
    # data = (data - data_min) / (data_max - data_min)
    labels = torch.cat(labels).cpu().numpy().tolist()
    sets = set(labels)
    dict = {}
    for item in sets:
        dict.update({item: labels.count(item)})

    # labels = torch.cat(gt_list[:80])
    # print(X.shape)
    # print(X.shape, label.shape)
    from torch.nn import functional as F
    data = F.normalize(data, dim=1)
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    with torch.no_grad():
        Y = tsne(data, 2, 256, 20.0)

    if 1:
        Y = Y.cpu().numpy()
    #matplot
    # pyplot.axis('off')
    # pyplot.xticks([])
    # pyplot.yticks([])
    # pyplot.scatter(Y[:, 0], Y[:, 1], 10, labels)
    # pyplot.show()
    # pyplot.savefig('tsne.png')
    #seaborn
    class_num = len(np.unique(labels))
    df = pd.DataFrame()
    df["y"] = labels
    df["comp1"] = Y[:, 0]
    df["comp2"] = Y[:, 1]

    sns.scatterplot(x=df.comp1.tolist(), y=df.comp2.tolist(), hue=df.y.tolist(), style=df.y.tolist(),
                    palette=sns.color_palette("Set2", class_num),
                    data=df).set(title="T-SNE projection")

    pyplot.axis('off')
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.savefig('tsne10.jpg', format="jpg")
    pyplot.show()
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("fcos_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    """

    :rtype:
    """
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("fcos_core.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(cfg, model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
