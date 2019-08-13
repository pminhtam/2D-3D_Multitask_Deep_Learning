import os,glob
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())


import deephar

from deephar.config import pennaction_dataconf

from deephar.models import reception
from deephar.models import action_2D as action
from deephar.utils import *
import numpy as np

num_frames = 4
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


# weights_path = "/mnt/hdd10tb/Users/pminhtamnb/deephar/weights_merlaction_new_050-0.9399.h5"
weights_path = "/mnt/hdd10tb/Users/pminhtamnb/deephar/weights_merlaction_new_4_010-0.8464.h5"
# weights_path = "weights_merlaction_3_006.h5"
model.load_weights(weights_path)

import cv2



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
        return img,bbox
    except:
        warning('Error loading sample key: %d' % (data))
        raise

def calc_dis_bbox(bbox1,bbox2):
    x1 = (bbox1[0]+bbox1[2])/2.0
    y1 = (bbox1[1]+bbox1[3])/2.0
    x2 = (bbox2[0] + bbox2[2]) / 2.0
    y2 = (bbox2[1] + bbox2[3]) / 2.0
    x = abs(x1 - x2)
    y = abs(y1 - y2)
    return np.sqrt(x**2+y**2)
import glob, os
os.chdir("./images")
for file in glob.glob("*.txt"):
    # print(file)
    data_path = "/home/pminhtamnb/proj4/deephar/images/"+file
    # data_path = "/home/andang/AlphaPose/human-detection/output/BBOX/1_1_crop.txt"
    # data_path = "/home/andang/AlphaPose/human-detection/output/BBOX/10_1_crop.txt.txt"
    # data_path = "/home/andang/AlphaPose/human-detection/output/BBOX/GlanceatShelf_train.txt"
    f = open(data_path)
    datas = []
    for i in f:
        # print(i)
        datas.append(i.strip("\n").split("\t"))
    # print(data)
    f.close()
    # i = 7
    i = 3
    classes = ["ApproachtoShelf","GlanceAtShelf","LookatShelf"]

    while True:
        if i >= len(datas):
        # if i >= 100:
            break
        # data1 = datas[i-7]
        # data2 = datas[i-6]
        # data3 = datas[i-5]
        # data4 = datas[i-4]
        data5 = datas[i-3]
        data6 = datas[i-2]
        data7 = datas[i-1]
        data8 = datas[i]
        # img1,bbox1 = load_image(data1)
        # img2,bbox2 = load_image(data2)
        # img3,bbox3 = load_image(data3)
        # img4,bbox4 = load_image(data4)
        img5,bbox5 = load_image(data5)
        img6,bbox6 = load_image(data6)
        img7,bbox7 = load_image(data7)
        img8,bbox8 = load_image(data8)

        # dis = calc_dis_bbox(bbox1,bbox8)    # tinh khoang cach giua 2 bbox
        dis = calc_dis_bbox(bbox5,bbox8)    # tinh khoang cach giua 2 bbox

        # video = [img1,img2,img3,img4,img5,img6,img7,img8]
        video = [img5,img6,img7,img8]
        pred = model.predict(np.expand_dims(video, axis=0))
        # print(pred)

        print(i,"    ", classes[np.argmax(pred)])
        # print("{} {}   {}  {}    {}".format(i,i+1,i+2,i+3,classes[np.argmax(pred[-3])]))

        k = i
        fontScale = 1

        fontColor = (255, 0, 0)
        img = cv2.imread(datas[k][0])
        cv2.rectangle(img,(bbox8[0],bbox8[1]),(bbox8[2],bbox8[3]),(0,0,255),2)
        if (dis > 20):
            cv2.putText(img, "GlanceAtShelf" , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontColor,
                        thickness=3)
        else:
            # cv2.putText(img,classes[np.argmax(pred[-3])] ,(50,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale,fontColor,thickness = 3)
            cv2.putText(img,classes[np.argmax(pred)] ,(50,50),cv2.FONT_HERSHEY_SIMPLEX,fontScale,fontColor,thickness = 3)

        cv2.putText(img,"ApproachtoShelf: " + str(pred[0][0]), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontColor,thickness = 3)
        cv2.putText(img, "GlanceAtShelf: " + str(pred[0][1]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontColor,thickness = 3)
        cv2.putText(img, "LookatShelf: " + str(pred[0][2]), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontColor,thickness = 3)
        # cv2.imwrite("zb"+'{:04d}'.format(k)+".jpg",img)
        # cv2.imwrite("zb"+'{:04d}'.format(k)+".jpg",img)
        cv2.imwrite("/mnt/hdd10tb/Users/pminhtamnb/img_out_4/"+file+'{:04d}'.format(k)+".jpg",img)

        i = i + 1