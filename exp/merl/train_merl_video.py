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
from deephar.models import action_pose as action
from deephar.utils import *
from deephar.callbacks import SaveModel

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))


sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

annothelper.check_pennaction_dataset()

num_frames = 16
use_bbox = False
# use_bbox = False
num_blocks = 4
batch_size = 2
input_shape = pennaction_dataconf.input_shape
num_joints = 17
num_actions = 2


model_pe = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5))


model = action.build_merge_model(model_pe, num_actions, input_shape,
        num_frames, num_joints, num_blocks, pose_dim=2, pose_net_version='v1',
        full_trainable=False)
# model.summary()


