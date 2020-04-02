## Requirements
* Python 3.6
* [Pytorch](https://pytorch.org) >= 1.2.0 
* python-opencv
* [py-motmetrics](https://github.com/cheind/py-motmetrics) (`pip install motmetrics`)
* cython-bbox (`pip install cython_bbox`)
* (Optional) ffmpeg (used in the video demo)
* (Optional) [syncbn](https://github.com/ytoon/Synchronized-BatchNorm-PyTorch) (compile and place it under utils/syncbn, or simply replace with nn.BatchNorm [here](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/models.py#L12))
* [Apex](https://github.com/NVIDIA/apex)



Usage:
```
python demo.py --input-video path/to/your/input/video --weights path/to/model/weights
               --output-format video --output-root path/to/output/root --batch-size n
```

## Pretrained model and baseline models
Darknet-53 ImageNet pretrained model: [[DarkNet Official]](https://pjreddie.com/media/files/darknet53.conv.74)


## Test on MOT-16 Challenge
```
python track.py --cfg ./cfg/yolov3_1088x608.cfg --weights /path/to/model/weights
```
By default the script runs evaluation on the MOT-16 training set. If you want to evaluate on the test set, please add `--test-mot16` to the command line.
Results are saved in text files in `$DATASET_ROOT/results/*.txt`. You can also add `--save-images` or `--save-videos` flags to obtain the visualized results. Visualized results are saved in `$DATASET_ROOT/outputs/`


