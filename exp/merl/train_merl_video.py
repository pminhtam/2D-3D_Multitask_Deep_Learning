import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from keras.utils.data_utils import get_file

from deephar.config import pennaction_dataconf

from deephar.data import MERLAction
from deephar.data import BatchLoader

from deephar.models import reception
from deephar.models import action_2D as action
from deephar.utils import *
from deephar.callbacks import SaveModel

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))


sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

annothelper.check_pennaction_dataset()

"""
num_frames = 16
use_bbox = False
# use_bbox = False
num_blocks = 4
batch_size = 2
input_shape = pennaction_dataconf.input_shape
num_joints = 16
num_actions = 15

model_pe = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5),
        concat_pose_confidence=False)

model = action.build_merge_model(model_pe, num_actions, input_shape,
        num_frames, num_joints, num_blocks, pose_dim=2, pose_net_version='v1',
        full_trainable=False)

model.summary()

"""

num_frames = 3
use_bbox = False
# use_bbox = False
num_blocks = 4
batch_size = 2
input_shape = pennaction_dataconf.input_shape
num_joints = 16
num_actions = 2


model_pe = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5))
weights_path = ""
model_pe.load_weights(weights_path)
model = action.build_merge_model(model_pe, num_actions, input_shape,
        num_frames, num_joints, num_blocks, pose_dim=2, pose_net_version='v1',
        full_trainable=False)
model.summary()
# """

# """
anno_path = "/home/pminhtamnb/proj4/7-kpts/merl4000_4300.pkl"
dataset_path = "/mnt/hdd10tb/Users/duong/MERL"
merl_seq = MERLAction(dataset_path,anno_path, pennaction_dataconf,
        poselayout=pa16j2d, clip_size=num_frames)

merl_te = BatchLoader(merl_seq, ['frame'], ['merlaction'], TRAIN_MODE,
        batch_size=1, shuffle=False,num_predictions=11)

callbacks = []
weights_file = 'weights_merlaction_{epoch:03d}.h5'

callbacks.append(SaveModel(weights_file))

model.fit_generator(merl_te,
# model.fit(x,y,
#         steps_per_epoch=None,
        epochs=10,
        callbacks=callbacks,
        workers=4,
        initial_epoch=0)
# """