from __future__ import print_function
import multiprocessing
from multiprocessing import Process, Manager, Queue
import logging
import argparse
import motmetrics as mm
import copy
import torch
from tracker.multitracker import JDETracker
from utils import visualization as vis
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
from utils.parse_config import parse_model_cfg
import utils.datasets as datasets
import csv
from utils.utils import *
import multiprocessing as mp
import traceback

mpl = multiprocessing.log_to_stderr()
mpl.setLevel(logging.INFO)



def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def post_proc(queue, state, tracker, results, save_dir, show_image, opt, result_filename, data_type, frame_id):
    while True:
        lisst = queue.get()
        predslist, img0list, _ = lisst[0], lisst[1], lisst[2]  # Read from the queue and do nothing
        if img0list == 0:
            write_results(result_filename, results, data_type)

            break

        for preds, img0 in zip(predslist, img0list):
            state, frame_id = tracker.workOnDetections(opt, preds, results, img0, frame_id, save_dir, show_image, state)




def write_preds(preds, img0, frame_id, queue):
    listt = [preds, img0, frame_id]
    queue.put(listt)


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=False, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = {'0': 0}
    itr_id = 0
    self_dict = copy.deepcopy(tracker.__dict__)
    del self_dict['model']
    del self_dict['opt']

    pqueue = Queue()  # writer() writes to pqueue from _this_ process
    reader_p = Process(target=post_proc, args=((pqueue, self_dict, tracker, results, save_dir, show_image, opt, result_filename, data_type, frame_id)))
    reader_p.daemon = True
    reader_p.start()

    for path, img, img0 in dataloader:
        # cv2.imwrite("test.jpeg", img0[0])
        if itr_id % 2 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(itr_id * opt.batch_size,
                                                                  float(opt.batch_size) / max(1e-5,
                                                                                              timer.average_time)))

        timer.tic()
        img = np.array(img)
        blob = torch.from_numpy(img).cuda()  # .unsqueeze(0)        # ChangeHere
        preds = tracker.getDetections(blob)
        # preds = torch.tensor(np.random.randn(10, 500, 518)).half()
        write_preds(preds, img0, frame_id, pqueue)
        # preds = preds.cpu()
        # preds_arr = preds.numpy()
        timer.toc()

        itr_id += 1
    write_preds(0, 0, 0, pqueue)

    return frame_id['0'], timer.average_time, timer.calls


def batch_size_effect_measure(opt, dataloader, data_type, result_filename, save_dir=None, show_image=False, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = {'0': 0}
    gpu_used = 0
    itr_id = 0
    self_dict = copy.deepcopy(tracker.__dict__)
    del self_dict['model']
    del self_dict['opt']

    pqueue = Queue()  # writer() writes to pqueue from _this_ process
    reader_p = Process(target=post_proc, args=(
    (pqueue, self_dict, tracker, results, save_dir, show_image, opt, result_filename, data_type, frame_id)))
    reader_p.daemon = True
    reader_p.start()

    for path, img, img0 in dataloader:
        # cv2.imwrite("test.jpeg", img0[0])
        if itr_id % 2 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(itr_id * opt.batch_size,
                                                                  float(opt.batch_size) / max(1e-5,
                                                                                              timer.average_time)))

        if itr_id == 0:
            gpu_used = int(get_gpu_memory_map()[0]) - 129
        if img == 0:
            break
        timer.tic()
        img = np.array(img)
        blob = torch.from_numpy(img).cuda()  # .unsqueeze(0)        # ChangeHere
        preds = tracker.getDetections(blob)
        # preds = torch.tensor(np.random.randn(10, 500, 518)).half()
        write_preds(preds, img0, frame_id, pqueue)
        # preds = preds.cpu()
        # preds_arr = preds.numpy()
        timer.toc()

        itr_id += 1

    fps = float(opt.batch_size) / max(1e-5, timer.average_time)
    write_preds(0, 0, 0, pqueue)

    # for proc in jobs:
    #     proc.join()
    # save results
    return fps, gpu_used


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    """Run demo.py for accurate fps info"""
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # Read config
    cfg_dict = parse_model_cfg(opt.cfg)
    opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None

        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='track.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--test-mot16', action='store_true', help='tracking buffer')
    parser.add_argument('--save-images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save-videos', action='store_true', help='save tracking results (video)')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    if not opt.test_mot16:
        seqs_str = '''MOT16-02
             MOT16-04
             MOT16-05
             MOT16-09
             MOT16-10
             MOT16-11
             MOT16-13
            '''
        data_root = '/home/nisarg/guard/Custom_data/MOT16/train'
    else:
        seqs_str = '''MOT16-01
                     MOT16-03
                     MOT16-06
                     MOT16-07
                     MOT16-08
                     MOT16-12
                     MOT16-14'''
        data_root = '/home/wangzd/datasets/MOT/MOT16/images/test'

    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.weights.split('/')[-2],
         show_image=False,
         save_images=opt.save_images,
         save_videos=opt.save_videos)

