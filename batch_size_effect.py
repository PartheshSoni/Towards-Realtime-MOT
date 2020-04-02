"""Demo file for running the JDE tracker on custom video sequences for pedestrian tracking.

This file is the entry point to running the tracker on custom video sequences. It loads images from the provided video sequence, uses the JDE tracker for inference and outputs the video with bounding boxes indicating pedestrians. The bounding boxes also have associated ids (shown in different colours) to keep track of the movement of each individual.

Examples:
        $ python demo.py --input-video path/to/your/input/video --weights path/to/model/weights --output-root path/to/output/root


Attributes:
    input-video (str): Path to the input video for tracking.
    output-root (str): Output root path. default='results'
    weights (str): Path from which to load the model weights. default='weights/latest.pt'
    cfg (str): Path to the cfg file describing the model. default='cfg/yolov3.cfg'
    iou-thres (float): IOU threshold for object to be classified as detected. default=0.5
    conf-thres (float): Confidence threshold for detection to be classified as object. default=0.5
    nms-thres (float): IOU threshold for performing non-max supression. default=0.4
    min-box-area (float): Filter out boxes smaller than this area from detections. default=200
    track-buffer (int): Size of the tracking buffer. default=30
    output-format (str): Expected output format, can be video, or text. default='video'


Todo:
    * Add compatibility for non-GPU machines (would run slow)
    * More documentation
"""

import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import csv
from tracker.multitracker import JDETracker
from utils import visualization as vis
from utils.utils import *
from utils.io import read_results
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
from utils.parse_config import parse_model_cfg
import utils.datasets as datasets
import time
import torch
from track import batch_size_effect_measure

logger.setLevel(logging.INFO)


def track(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    cfg_dict = parse_model_cfg(opt.cfg)
    opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]
    batch_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    batch_list
    gpu_used_list = []
    fps_list = []

    for batch in batch_list:
        # run tracking
        opt.batch_size = batch
        timer = Timer()
        accs = []
        n_frame = 0

        logger.info('Starting processing for batch_size: {}'.format(batch))
        dataloader = datasets.LoadVideoBatches(opt.input_video, opt.batch_size, opt.img_size)
        result_filename = os.path.join(result_root, 'results.txt')
        frame_rate = dataloader.frame_rate

        frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
        # try:
        fps, gpu_used = batch_size_effect_measure(opt, dataloader, 'mot', result_filename,
                                                  save_dir=None, show_image=False, frame_rate=frame_rate)
        gpu_used_list.append(gpu_used)
        fps_list.append(fps)

    filename_for_analsis = "Batch_analysis_results/results_" + str(opt.video_number)+".csv"
    with open(filename_for_analsis, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Batch_size", "GPU_used", "fps"])

        print("\n\n\n")
        print("Batch_size   GPU_used   fps")
        for i in range(len(batch_list)):
            print(batch_list[i], "   ", gpu_used_list[i], "   ", fps_list[i])
            writer.writerow([batch_list[i], gpu_used_list[i], fps_list[i]])

    import matplotlib.pyplot as plt

    # x axis values
    x = batch_list
    # corresponding y axis values
    y = gpu_used_list

    # plotting the points
    plt.plot(x, y, label="GPU usage")

    # naming the x axis
    plt.xlabel('Batch size')
    # naming the y axis
    plt.ylabel('GPU Utilization (in MB)')

    # giving a title to my graph
    plt.title('GPU usage for varying batch sizes')
    plt.legend(loc="upper right")
    plt.grid()

    # function to show the plot
    graphname = "GPU_graphs/GPU_"+str(opt.video_number)+".svg"
    plt.savefig(graphname)

    plt.clf()
    plt.cla()
    plt.close()

    # x axis values
    x = batch_list
    # corresponding y axis values
    y = fps_list

    # plotting the points
    plt.plot(x, y, label="Speed")

    # naming the x axis
    plt.xlabel('Batch size')
    # naming the y axis
    plt.ylabel('Speed (in FPS)')

    # giving a title to my graph
    plt.title('Speed for different batch sizes')
    plt.legend(loc="upper right")
    plt.grid()

    # function to show the plot
    graphname = "FPS_graphs/FPS_"+str(opt.video_number)+".svg"
    plt.savefig(graphname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='demo.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--input-video', type=str, help='path to the input video')
    parser.add_argument('--output-format', type=str, default='video', choices=['video', 'text'],
                        help='Expected output format. Video or text.')
    parser.add_argument('--output-root', type=str, default='results', help='expected output root path')
    parser.add_argument('--batch-size', type=int, default='1', help='Batch size for feeding the model')
    parser.add_argument('--video-number', type=int, default='1', help='It indicates the number of video sequence to '
                                                                      'save properly')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    track(opt)
