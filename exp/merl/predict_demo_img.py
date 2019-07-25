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

sys.path.append(os.path.join(os.getcwd(), 'datasets'))



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

# """
"""Load pre-trained model."""
# """
weights_path = "weights_merl_061.h5"
model.load_weights(weights_path)


import cv2



# data_path = "/home/andang/AlphaPose/human-detection/output/BBOX/1_1_crop.txt"
data_path = "/home/andang/AlphaPose/human-detection/output/BBOX/10_1_crop.txt.txt"

f = open(data_path)
datas = []
for i in f:
    # print(i)
    datas.append(i.strip("\n").split("\t"))
# print(data)
f.close()

def load_image(data):
    img_path = data[0]
    bbox = [int(data[2]),int(data[3]),int(data[4]),int(data[5])]
    try:
        imgt = cv2.imread(img_path) / 255.0
        img = np.zeros((int(round(bbox[3])), int(round(bbox[2]))))
        try:  # crop image with bbox
            img = imgt[int(bbox[1]):int(round(bbox[3])), \
                  int(bbox[0]):int(round(bbox[2]))]  # crop image
        except:
            print(bbox)
            print("error   : img  ", img_path)

        img = cv2.resize(img, (256, 256))
        return img
    except:
        warning('Error loading sample key: %d' % (data))
        raise



import matplotlib.pyplot as plt
import numpy as np
k = 0
for i in datas:

    img = load_image(i)
    plt.axis("off")
    # print(img.shape)
    plt.imshow(img)

    pred = model.predict(np.array([img]))

    for j in range(7,8):
        for zz in pred[j][0]:

            if zz[2]>0.5:
                plt.scatter(zz[0] * 256, zz[1] * 256)
    # plt.savefig("zia"+'{:04d}'.format(k)+".jpg")
    print(k)
    plt.savefig("zib"+'{:04d}'.format(k)+".jpg")
    k+=1
    plt.clf()


