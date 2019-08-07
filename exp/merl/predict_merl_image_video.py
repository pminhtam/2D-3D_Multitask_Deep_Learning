import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())
from keras.models import Model
from keras.layers import concatenate
from deephar.config import pennaction_dataconf
from deephar.config import mpii_sp_dataconf

from deephar.data import MERL5Action
from deephar.models import reception

from deephar.utils import *

import numpy as np
import argparse
import matplotlib.pyplot as plt


"""Architecture configuration."""
num_frames = 8
num_blocks = 8
input_shape = mpii_sp_dataconf.input_shape
num_joints = 16

"""Build pose and action models."""
model = reception.build(input_shape, num_joints, dim=2,
                           num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5),
                           concat_pose_confidence=False)
outputs = []
for b in range(int(len(model.outputs) / 2)):
    outputs.append(concatenate([model.outputs[2*b], model.outputs[2*b + 1]],
        name='blk%d' % (b + 1)))
model = Model(model.input, outputs, name=model.name)



val_anno_path = "/mnt/hdd10tb/Users/andang/actions/test_2.json"
val_merl_seq = MERL5Action(val_anno_path, pennaction_dataconf,
                           poselayout=pa16j2d, clip_size=num_frames)

weights_path = "weights_merl_061.h5"
model.load_weights(weights_path)
classes = {0:"ApproachtoShelf",1:"GlanceAtShelf",2:"LookatShelf"}
for i in range(20):
    data = val_merl_seq.get_data(i)
    frames = data['frame']
    action = data['merlaction']

    # print(i)
    # print(len(frames))
    for j in range(len(frames)):
        frame = frames[j]
        pred= model.predict(np.array([frame]))
        plt.imshow(frame)
        for zz in pred[7][0]:
            # for zz in pred:
            # print(zz)
            if zz[2] > 0.3:
                plt.scatter(zz[0] * 256, zz[1] * 256)
        plt.savefig("merl3_{}_idx_{:03d}_frame_{:02d}.jpg".format(classes[np.argmax(action)],i,j))
        plt.clf()
