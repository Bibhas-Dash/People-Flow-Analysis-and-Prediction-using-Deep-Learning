# People-Flow-Analysis-and-Prediction-using-Deep-Learning
Here we have tried to estimate the concentration and behavior of a crowd in the videos of observers and forecast the expected congregation in the ROI (region of interest).

# Introduction:
People detection and flow analysis of crowd from visual data of congregation scenes have a wide range of application in areas such as urban planning, customer behavior assessment, anomaly detection, public safety and visual surveillance.The purpose of people flow analysis is to estimate the concentration and behavior of a crowd in the videos of observers and forecast the expected congregation in the ROI (region of interest).

# Requirements:
* Python 3.6
* Pytorch >= 1.2.0
* python-opencv
* py-motmetrics (pip install motmetrics)
* cython-bbox (pip install cython_bbox)

# Pretrained model and baseline models:
Darknet-53 ImageNet pretrained model: [DarkNet Official](https://pjreddie.com/media/files/darknet53.conv.74)

Trained models with different input resolutions:

Model	MOTA	IDF1	IDS	FP	FN	FPS	Link

JDE-1088x608	74.8	67.3	1189	5558	21505	22.2	[Google](https://drive.google.com/open?id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA)

JDE-864x480	70.8	65.8	1279	5653	25806	30.3	[Google](https://drive.google.com/open?id=1UKgkYrsV-59kYaHgWeJ70p5Mij3QWuFr)

JDE-576x320	63.7	63.3	1307	6657	32794	37.9	[Google](https://drive.google.com/file/d/1sca65sHMnxY7YJ89FJ6Dg3S3yAjbLdMz/view?usp=sharing)

Used Model- JDE-1088x608

# Usage:
python demo.py --input-video path/to/your/input/video --weights path/to/model/weights
               --output-format video --output-root path/to/output/root

# Acknowledgement:
A large portion of code is borrowed from [Zhongdao/Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT)
