import os
import sys
from keras.models import Model
from keras.layers import concatenate

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())


from deephar.config import mpii_sp_dataconf

from deephar.data import MERLSinglePerson


from deephar.models import reception
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from mpii_tools import eval_singleperson_pckh

sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

annothelper.check_mpii_dataset()


"""Architecture configuration."""
num_blocks = 8
batch_size = 24
input_shape = mpii_sp_dataconf.input_shape
num_joints = 16
# """
model = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5),
        concat_pose_confidence=False)
# """
"""Merge pose and visibility as a single output."""
# """
outputs = []
for b in range(int(len(model.outputs) / 2)):
    outputs.append(concatenate([model.outputs[2*b], model.outputs[2*b + 1]],
        name='blk%d' % (b + 1)))
model = Model(model.input, outputs, name=model.name)


weights_path = "weights_merl_061.h5"
model.load_weights(weights_path)
anno_path = "/home/pminhtamnb/proj4/7-kpts/merl4000_4300.pkl"
dataset_path = "/mnt/hdd10tb/Users/duong/MERL"
mpii = MERLSinglePerson(dataset_path,anno_path,dataconf=mpii_sp_dataconf)


import matplotlib.pyplot as plt
import numpy as np
# input = np.array(Image.open("000001.jpg").resize((256,256)))/255.0
# input = np.array(Image.open("aa.jpg").resize((256,256)))/255.0
# input = np.array(Image.open("datasets/MPII/images/069937887.jpg").resize((256,256)))/255.0
# input = np.array(Image.open("datasets/MPII/images/099946068.jpg").resize((256,256)))/255.0
# input = np.array(Image.open("/mnt/hdd10tb/Datasets/MERL_Shopping/ReachToShelf/31_2_crop_1150_1171_ReachToShelf-0005.jpg").resize((256,256)))/255.0
# input = np.array(Image.open("frame0.png").resize((256,256)))/255.0

data = mpii.get_data(5)
input = data['image']
label = data['pose']
# input = np.array([input])[:,:,:,:3]
print(label)
plt.imshow(input)
# plt.savefig("mpii_input")

pred = model.predict(np.array([input]))
# print(pred)

# for i in pred:
#     print(" predict  : ", i.shape)


# """
plt.imshow(input)
print(len(pred))
for j in range(7,8):
    for zz in pred[j][0]:
    # for zz in pred:
        print(zz)
        if zz[2]>0.5:
            plt.scatter(zz[0] * 256, zz[1] * 256)
plt.savefig("merl_1.jpg")
# """

