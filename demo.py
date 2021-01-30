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
import logging
import argparse
from utils.utils import *
from utils.log import logger
from utils.timer import Timer
from utils.parse_config import parse_model_cfg
import utils.datasets as datasets
from track import eval_seq
import matplotlib.pyplot as pp#bd{
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional#bd} 

logger.setLevel(logging.INFO)

def track(opt):
    result_root = opt.output_root if opt.output_root!='' else '.'
    mkdir_if_missing(result_root)

    cfg_dict = parse_model_cfg(opt.cfg)
    opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    # run tracking
    timer = Timer()
    accs = []
    n_frame = 0

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate 

    frame_dir = None if opt.output_format=='text' else osp.join(result_root, 'frame')
    try:
        ob=eval_seq(opt, dataloader, 'mot', result_filename,
                 save_dir=frame_dir, show_image=False, frame_rate=frame_rate)
        o=ob[3][1:]#bd{
        i=ob[4]
        l_count=0
        l_count=ob[5]
        ai=sum(i)/len(i)
        ao=sum(o)/len(o)
        print("\nAverage Inflow=", ai)
        print("\nAverage Outflow=", ao)
        if(ai>ao):
          print("\n                             Congratulations!! The ROI is proving to be a hotspot  !!!\n")
        elif(ao>ai):
          print("\n                             The ROI is losing people attention !!!\n")
        else:
          print("\n                             The ROI has a moderate people attention !!!\n")#bd}
        print("\n                             The ROI has a current crowd of %d people !!!\n"%l_count)
    except Exception as e:
        logger.info(e)
    
    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)
    
    return o,i,l_count

# split a univariate sequence
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

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
    parser.add_argument('--output-format', type=str, default='video', choices=['video', 'text'], help='Expected output format. Video or text.')
    parser.add_argument('--output-root', type=str, default='results', help='expected output root path')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    l=track(opt)#bd{
    o1=l[0]
    i1=l[1]
    l_count1=l[2]
    pp.style.use(['dark_background','seaborn-darkgrid'])
    pp.rcParams["figure.figsize"] = (20,3)
    pp.plot(o1,'r-o')
    pp.ylabel('Crowd Outflow from ROI')
    pp.show()
    print("\n")
    pp.plot(i1,'g-o')
    pp.ylabel('Crowd Inflow into ROI')
    pp.show()

    # define input sequence, here i1 and o1
    # choose a number of time steps
    n_steps = 3
    n_steps1 = 5
    # split into samples
    X, y = split_sequence(i1, n_steps)
    X1, y1 = split_sequence(o1, n_steps1)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    n_features1 = 1
    X1 = X1.reshape((X1.shape[0], X1.shape[1], n_features1))
    # define inflow model
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # define outflow model
    model1 = Sequential()
    model1.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps1, n_features1)))
    model1.add(Dense(1))
    model1.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=200, verbose=0)
    model.fit(X1, y1, epochs=200, verbose=0)
    # demonstrate prediction
    x_input = array(i1[-3:])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    x_input1 = array(o1[-5:])
    x_input1 = x_input1.reshape((1, n_steps1, n_features1))
    yhat1 = model.predict(x_input1, verbose=0)
    print('\nPredicted Crowd Inflow into ROI in upcoming time -->',round(yhat[0][0]))
    print('\nPredicted Crowd Outflow from ROI in upcoming time -->',round(yhat1[0][0]))
    pp.plot(yhat[0],'c-*')
    pp.ylabel('Predicted Crowd Inflow into ROI in upcoming time')
    pp.show()
    print("\n")
    pp.plot(yhat1[0],'y-*')
    pp.ylabel('Predicted Crowd Outflow from ROI in upcoming time')
    pp.show()
    print(l_count1)
    print('\nCrowd in ROI expected in upcoming time -->',round(l_count1-yhat1[0][0]+yhat[0][0]))#bd}

