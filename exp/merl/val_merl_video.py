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
parser.add_argument("--weights-file",type=str,default=None,help="weights path",required=True)
parser.add_argument("--label-path",type=str,default="/mnt/hdd10tb/Users/andang/actions/test_2.json",help="file json val label")
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

weights_path = args.weights_file
# weights_path = "/mnt/hdd10tb/Users/pminhtamnb/deephar/weights_merlaction_new_050-0.9399.h5"
# weights_path = "weights_merlaction_new_done.h5"
# weights_path = "weights_merlaction_1_041.h5"

model.load_weights(weights_path)
printcn(OKBLUE,"Load model done")

# val_anno_path = "/mnt/hdd10tb/Users/andang/actions/test_2.json"
# val_anno_path = "/mnt/hdd10tb/Users/andang/actions/train_2.json"
val_anno_path = args.label_path
val_merl_seq = MERL5Action(val_anno_path,pennaction_dataconf,
        poselayout=pa16j2d, clip_size=num_frames)
printcn(OKGREEN,"Load data test done")

classes = {"ApproachtoShelf":0,"GlanceAtShelf":1,"LookatShelf":2}
result_m = np.zeros((len(classes),len(classes)),dtype=int)
result_mmm = np.zeros((len(classes),len(classes)),dtype=int)
result_mmmvvv = np.zeros((len(classes),len(classes)),dtype=int)

from tqdm import tqdm
for i in tqdm(range(val_merl_seq.get_length())):
    data = val_merl_seq.get_data(i)
    pred = model.predict(np.expand_dims(data['frame'], axis=0))

    label = np.argmax(data['merlaction'])
    pred_m = np.argmax(pred[-3])
    pred_mmm = np.argmax(pred[-2])
    pred_mmmvvv = np.argmax(pred[-1])
    result_m[label][pred_m] +=1
    result_mmm[label][pred_mmm] +=1
    result_mmmvvv[label][pred_mmmvvv] +=1

print(result_m)
print(result_mmm)
print(result_mmmvvv)
