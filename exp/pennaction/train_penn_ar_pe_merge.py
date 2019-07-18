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
from deephar.models import action_2D as action
from deephar.utils import *
from deephar.callbacks import SaveModel

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

# weights_path = get_file(weights_file, TF_WEIGHTS_PATH, md5_hash=md5_hash,
#         cache_subdir='models')

# print("wwwww xxxxxxxx c zzz   ",weights_path)
# model.load_weights(weights_path)



"""Load PennAction dataset."""

penn_seq = PennAction('datasets/PennAction', pennaction_dataconf,
        poselayout=pa16j2d, topology='sequences', use_gt_bbox=use_bbox,
        clip_size=num_frames)

penn_te = BatchLoader(penn_seq, ['frame'], ['pennaction'], TRAIN_MODE,
        batch_size=1, shuffle=False,num_predictions=11)

print(penn_te.get_length(TRAIN_MODE))
# print(penn_te.get_length(TEST_MODE))
# data = penn_te.__getitem__(1)
# print(data)
# print(data[0])
# print(data[1])
# x = data[0]
# y = data[1]
# """
printcn(OKGREEN, 'Evaluation on PennAction multi-clip using predicted bboxes')


from keras.optimizers import SGD

lr = 0.001
momentum = 0.95
loss_weights=None
model.compile(loss='categorical_crossentropy',
                optimizer=SGD(lr=lr, momentum=momentum, nesterov=True),
                metrics=['acc'], loss_weights=loss_weights)
callbacks = []
weights_file = 'weights_pennaction_{epoch:03d}.h5'

callbacks.append(SaveModel(weights_file))

epochs=120
steps_per_epoch = penn_te.get_length(TRAIN_MODE)/epochs


import numpy as np

model.fit_generator(penn_te,
# model.fit(x,y,
#         steps_per_epoch=None,
        epochs=10,
        callbacks=callbacks,
        workers=4,
        initial_epoch=0)

# """