
Requirements:
Python 3.6
Pytorch >= 1.2.0
python-opencv
py-motmetrics (pip install motmetrics)
cython-bbox (pip install cython_bbox)
(Optional) ffmpeg (used in the video demo)
(Optional) syncbn (compile and place it under utils/syncbn, or simply replace with nn.BatchNorm here)
Apex (https://github.com/NVIDIA/apex)


Usage:

python demo.py --input-video path/to/your/input/video --weights path/to/model/weights
               --output-format video --output-root path/to/output/root --batch-size n
               
         
For Evaluation:
python track.py --cfg ./cfg/yolov3_1088x608.cfg --weights /path/to/model/weights

Please note that this is for MOT16 video sequences.






