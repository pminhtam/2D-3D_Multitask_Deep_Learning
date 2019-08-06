import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from keras.utils.data_utils import get_file

from deephar.config import pennaction_dataconf

from deephar.data import MERLAction
from deephar.data import MERL5Action
from deephar.data import BatchLoader

from deephar.models import reception
from deephar.models import action_2D as action
from deephar.utils import *
from deephar.callbacks import SaveModel
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--weights-file",type=str,default=None,help="weights path")
args = parser.parse_args()


"""Architecture configuration."""
num_frames = 8
use_bbox = False
# use_bbox = False
num_blocks = 8
batch_size = 2
input_shape = pennaction_dataconf.input_shape
num_joints = 16
num_actions = 3

"""Build pose and action models."""
model_pe = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5),
        # concat_pose_confidence=False)
        )

model = action.build_merge_model(model_pe, num_actions, input_shape,
        num_frames, num_joints, num_blocks, pose_dim=2, pose_net_version='v1',
        full_trainable=False)

# weights_path = args.weights_file
weights_path = "/mnt/hdd10tb/Users/pminhtamnb/deephar/weights_merlaction_new_050-0.9399.h5"
# weights_path = "weights_merlaction_new_done.h5"
# weights_path = "weights_merlaction_1_041.h5"

model.load_weights(weights_path)

# val_anno_path = "/mnt/hdd10tb/Users/andang/actions/test_2.json"
val_anno_path = "/mnt/hdd10tb/Users/andang/actions/train_2.json"
val_merl_seq = MERL5Action(val_anno_path,pennaction_dataconf,
        poselayout=pa16j2d, clip_size=num_frames)

total = 0
total_0 = 0
true_0 = 0
true_0_last = 0
true_1 = 0
true_1_last = 0
total_1 = 0
true_2 = 0
true_2_last = 0
total_2 = 0
true = 0
true_last = 0
from tqdm import tqdm
for i in tqdm(range(val_merl_seq.get_length())):
# for i in ddd:
    data = val_merl_seq.get_data(i)
    pred = model.predict(np.expand_dims(data['frame'], axis=0))
    # print(pred)
    # print(data['merlaction'])
    # """
    if np.argmax(data['merlaction']) == 0:
        total_0 += 1
        if np.argmax(pred[-3])==np.argmax(data['merlaction']):
            true_0+=1
            true+=1
        if np.argmax(pred[-1])==np.argmax(data['merlaction']):
            true_0_last+=1
            true_last+=1
    elif np.argmax(data['merlaction']) == 1:
        total_1 += 1
        if np.argmax(pred[-3])==np.argmax(data['merlaction']):
            true_1+=1
            true+=1
        if np.argmax(pred[-3])==np.argmax(data['merlaction']):
            true_1_last+=1
            true_last+=1
    elif np.argmax(data['merlaction']) == 2:
        total_2 += 1
        if np.argmax(pred[-3])==np.argmax(data['merlaction']):
            true_2+=1
            true+=1
        if np.argmax(pred[-3])==np.argmax(data['merlaction']):
            true_2_last+=1
            true_last+=1
    total +=1
    

print("total    ",total)
print("total_0    ",total_0)
print("true_0    ",true_0)
print("true_0_last    ",true_0_last)
print("total_1     ",total_1)
print("true_1     ",true_1)
print("true_1_last     ",true_1_last)
print("total_2     ",total_2)
print("true_2     ",true_2)
print("true_2_last     ",true_2_last)
print("true       ",true)
print("true_last       ",true_last)
# """
#111  -1
# total     3331
# true_0     44
# true_1      1598
# true        1642

#121  -1
# total     3331
# total_0     1710
# true_0     387
# total_1      1621
# true_1      1536
# true        1923

#121    -3
# total     3331
# total_0     1710
# true_0     587
# total_1      1621
# true_1      1451
# true        2038

#141
# total     3331
# total_0     1710
# true_0     1679
# true_0_last     1664
# total_1      1621
# true_1      1592
# true_1_last      1592
# true        3271
# true_last        3256
