import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from keras.models import Model
from keras.layers import concatenate
from keras.utils.data_utils import get_file

from deephar.config import mpii_sp_dataconf

from deephar.data import MpiiSinglePerson
from deephar.data import BatchLoader

from deephar.models import reception
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from mpii_tools import eval_singleperson_pckh

sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

annothelper.check_mpii_dataset()

weights_file = 'weights_PE_MPII_cvpr18_19-09-2017.h5'
TF_WEIGHTS_PATH = \
        'https://github.com/dluvizon/deephar/releases/download/v0.1/' \
        + weights_file
md5_hash = 'd6b85ba4b8a3fc9d05c8ad73f763d999'

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

"""Architecture configuration."""
num_blocks = 8
batch_size = 24
input_shape = mpii_sp_dataconf.input_shape
num_joints = 16

model = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5),
        concat_pose_confidence=False)

"""Load pre-trained model."""
weights_path = get_file(weights_file, TF_WEIGHTS_PATH, md5_hash=md5_hash,
        cache_subdir='models')
model.load_weights(weights_path)
# model.compile('sgd','mse')

"""Merge pose and visibility as a single output."""
# """
outputs = []
for b in range(int(len(model.outputs) / 2)):
    outputs.append(concatenate([model.outputs[2*b], model.outputs[2*b + 1]],
        name='blk%d' % (b + 1)))
# """
# model = Model(model.input, outputs, name=model.name)



import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# input = np.array(Image.open("000001.jpg").resize((256,256)))/255.0
# input = np.array(Image.open("aa.jpg").resize((256,256)))/255.0
# input = np.array(Image.open("datasets/MPII/images/069937887.jpg").resize((256,256)))/255.0
# input = np.array(Image.open("datasets/MPII/images/099946068.jpg").resize((256,256)))/255.0
input = np.array(Image.open("/mnt/hdd10tb/Datasets/MERL_Shopping/ReachToShelf/31_2_crop_1150_1171_ReachToShelf-0005.jpg").resize((256,256)))/255.0
# input = np.array(Image.open("frame0.png").resize((256,256)))/255.0
input = np.array([input])[:,:,:,:3]

plt.imshow(input[0])
# plt.savefig("mpii_input")

pred = model.predict(input)
# print(pred)

for i in pred:
    print(" predict  : ", i.shape)


# """
print(len(pred))
for j in range(7,8):
    for zz in pred[j][0]:
        print(zz)
        plt.scatter(zz[0] * 256, zz[1] * 256)
plt.savefig("mpii_pred_mpii_31_2_crop_1150_1171_ReachToShelf-0005.jpg")
# """

