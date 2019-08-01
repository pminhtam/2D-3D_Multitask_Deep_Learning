import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from keras.utils.data_utils import get_file

from deephar.config import pennaction_dataconf


from deephar.models import reception
from deephar.models import action_pose as action


sys.path.append(os.path.join(os.getcwd(), 'exp/common'))


sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

# annothelper.check_pennaction_dataset()


num_frames = 8
use_bbox = False
# use_bbox = False
num_blocks = 8
batch_size = 4
input_shape = pennaction_dataconf.input_shape
num_joints = 16
num_actions = 3



model = action.build_merge_model(num_actions, input_shape,
        num_frames, num_joints, pose_dim=2, pose_net_version='v1',
        full_trainable=False)

# weights_path = "weights_merlaction_1_003.h5"
# model.load_weights(weights_path)
# model.summary()
# """
