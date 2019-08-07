import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())


import deephar

from deephar.config import pennaction_dataconf

from deephar.models import reception
from deephar.models import action_2D as action
from deephar.utils import *
from deephar.callbacks import SaveModel
import numpy as np

num_frames = 8
use_bbox = False
# use_bbox = False
num_blocks = 8
batch_size = 2
input_shape = pennaction_dataconf.input_shape
num_joints = 16
num_actions = 3


model_pe = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5),
        concat_pose_confidence=False)

model = action.build_merge_model(model_pe, num_actions, input_shape,
        num_frames, num_joints, num_blocks, pose_dim=2, pose_net_version='v1',
        full_trainable=False)


weights_path = "/mnt/hdd10tb/Users/pminhtamnb/deephar/weights_merlaction_new_050-0.9399.h5"
# weights_path = "weights_merlaction_3_006.h5"
model.load_weights(weights_path)

import pandas as pd
import cv2

# data_path = "/home/andang/AlphaPose/human-detection/output/BBOX/1_1_crop.txt"
data_path = "/home/andang/AlphaPose/human-detection/output/BBOX/10_1_crop.txt.txt"
# data_path = "/home/andang/AlphaPose/human-detection/output/BBOX/GlanceatShelf_train.txt"
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

i = 7
classes = ["ApproachtoShelf","GlanceAtShelf","LookatShelf"]

while True:
    if i >= len(datas):
    # if i >= 100:
        break
    data1 = datas[i-7]
    data2 = datas[i-6]
    data3 = datas[i-5]
    data4 = datas[i-4]
    data5 = datas[i-3]
    data6 = datas[i-2]
    data7 = datas[i-1]
    data8 = datas[i]
    img1 = load_image(data1)
    img2 = load_image(data2)
    img3 = load_image(data3)
    img4 = load_image(data4)
    img5 = load_image(data5)
    img6 = load_image(data6)
    img7 = load_image(data7)
    img8 = load_image(data8)
    video = [img1,img2,img3,img4,img5,img6,img7,img8]
    pred = model.predict(np.expand_dims(video, axis=0))
    print(i,"    ", classes[np.argmax(pred[-3])])
    # print("{} {}   {}  {}    {}".format(i,i+1,i+2,i+3,classes[np.argmax(pred[-3])]))

    k = i
    fontScale = 1

    fontColor = (255, 0, 0)
    # print(pred[-3])
    img = cv2.imread(datas[k][0])
    cv2.putText(img,classes[np.argmax(pred[-3])] ,(50,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale,fontColor,thickness = 3)
    cv2.putText(img,"ApproachtoShelf: " + str(pred[-3][0][0]), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontColor,thickness = 3)
    cv2.putText(img, "GlanceAtShelf: " + str(pred[-3][0][1]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontColor,thickness = 3)
    cv2.putText(img, "LookatShelf: " + str(pred[-3][0][2]), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontColor,thickness = 3)
    # cv2.imwrite("zb"+'{:04d}'.format(k)+".jpg",img)
    cv2.imwrite("zb"+'{:04d}'.format(k)+".jpg",img)

    i = i + 1