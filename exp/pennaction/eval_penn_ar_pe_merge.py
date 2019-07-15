import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from keras.utils.data_utils import get_file

from deephar.config import pennaction_dataconf

from deephar.data import PennAction
from deephar.data import BatchLoader

from deephar.models import reception
from deephar.models import action
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from penn_tools import eval_singleclip_generator
from penn_tools import eval_multiclip_dataset

sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

annothelper.check_pennaction_dataset()

weights_file = 'weights_AR_merge_ep074_26-10-17.h5'
TF_WEIGHTS_PATH = \
        'https://github.com/dluvizon/deephar/releases/download/v0.3/' \
        + weights_file
md5_hash = 'f53f89257077616a79a6c1cd1702d50f'

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')


num_frames = 16
use_bbox = False
# use_bbox = False
num_blocks = 4
batch_size = 2
input_shape = pennaction_dataconf.input_shape
num_joints = 16
num_actions = 15

"""Build pose and action models."""
model_pe = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5),
        concat_pose_confidence=False)

model = action.build_merge_model(model_pe, num_actions, input_shape,
        num_frames, num_joints, num_blocks, pose_dim=2, pose_net_version='v1',
        full_trainable=False)

"""Load pre-trained model."""

weights_path = get_file(weights_file, TF_WEIGHTS_PATH, md5_hash=md5_hash,
        cache_subdir='models')

print("wwwww xxxxxxxx c zzz   ",weights_path)
model.load_weights(weights_path)



"""Load PennAction dataset."""

penn_seq = PennAction('datasets/PennAction', pennaction_dataconf,
        poselayout=pa16j2d, topology='sequences', use_gt_bbox=use_bbox,
        clip_size=num_frames)

penn_te = BatchLoader(penn_seq, ['frame'], ['pennaction'], TEST_MODE,
        batch_size=1, shuffle=False)


printcn(OKGREEN, 'Evaluation on PennAction multi-clip using predicted bboxes')
# eval_multiclip_dataset(model, penn_seq,
#         bboxes_file='datasets/PennAction/penn_pred_bboxes_16f.json',
#         logdir=logdir)


# =================================================================
import numpy as np
import json
import matplotlib.pyplot as plt
frame_list = penn_seq.get_clip_index(0, TEST_MODE, subsamples=[2])
with open('datasets/PennAction/penn_pred_bboxes_16f.json', 'r') as fid:
    bboxes_data = json.load(fid)

i = 0
subsampling = 2
f = 0
hflip = 0
penn_seq.dataconf.fixed_hflip = hflip
key = '%04d.%d.%03d.%d' % (i, subsampling, f, hflip)
bbox = np.array(bboxes_data[key])
penn_seq.use_gt_bbox = False

data = penn_seq.get_data(0, TEST_MODE, frame_list=frame_list[0],
                            bbox=bbox)  # get image with bbox . crop image by bbox

actions_lb = ['baseball_pitch', 'baseball_swing', 'bench_press', 'bowl', 'clean_and_jerk',
              'golf_swing', 'jump_rope', 'jumping_jacks' ,'pullup', 'pushup', 'situp',\
                 'squat' ,'strum_guitar' ,'tennis_forehand' ,'tennis_serve']
# print(data['frame'])
# print(data['frame'].shape) # (16, 256, 256, 3)
# print(data['seq_idx'])
# print(data['frame_list'])
# print(data['pennaction'])
# for i in range(16):
#     plt.imshow((data['frame'][i]/2.+.5))
#     plt.savefig("frame"+str(i)+".png")

pred = model.predict(np.expand_dims(data['frame'], axis=0))
"""
print(len(pred))
for i in pred:

    print("predict  : ",np.argmax(i))
    print("predict  : ",actions_lb[np.argmax(i)])
# print("predict :  ", np.argmax(pred,axis=-1))
"""
print(pred[-1])
"""

from keras.utils import plot_model


# model_pe.summary()
# model.summary()
plot_model(model_pe,"model_pe.png",show_shapes=True,show_layer_names=True)
# plot_model(model,"model.png",show_shapes=True,show_layer_names=True)

"""




# CUDA_VISIBLE_DEVICES=0 python exp/pennaction/eval_penn_ar_pe_merge.py