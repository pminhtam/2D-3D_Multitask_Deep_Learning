import copy
import json
import math
import os
import pickle
import matplotlib.pyplot as plt

import cv2
import numpy as np
from pycocotools.coco import COCO
from PIL import Image,ImageDraw



ann_file = "/mnt/hdd10tb/Users/tam/COCO/annotations/person_keypoints_val2017.json"
image = "/mnt/hdd10tb/Users/tam/COCO/val2017"
# f =  open(ann_file, 'rb')
# coco = json.load(f)
# f.close()
coco = COCO(ann_file)
# print(coco.getImgIds())

# 15335     [1216332, 1230490, 1248228, 1277736, 1282318, 1318954, 1694331, 1705756, 1753431, 2025828, 2028889]
# 549220     [189158, 200727, 1716649, 1737443]
# 226154     [183110, 191567, 1228544, 2015782]
# for idx in coco.getImgIds():
#     annIds = coco.getAnnIds(idx)
#     print(idx,"   ",annIds)
    # print(annIds)

# anno = [1216332, 1230490, 1248228, 1277736, 1282318, 1318954, 1694331, 1705756, 1753431, 2025828, 2028889]
anno = [183110, 191567, 1228544, 2015782]

print(np.array(coco.loadAnns(441861)[0]['bbox']))
# print(np.array(coco.loadAnns(2164128)[0]['keypoints']).reshape(17,3))

"""
img = Image.open("/mnt/hdd10tb/Users/tam/COCO/val2017/000000226154.jpg")
# img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
ii = 0
cropped = []
for i in anno:
    ann = coco.loadAnns(i)[0]
    key_point = np.array(ann['keypoints']).reshape(17, 3)
    bbox = ann['bbox']
    # print(bbox)
    # cropped = imageObject.crop((100, 30, 400, 300))
    cropped = img.crop((bbox[0],bbox[1],bbox[2]+bbox[0],bbox[3]+bbox[1]))

    draw = ImageDraw.Draw(cropped)
    for k in key_point:
        if k[2]!=0:
            draw.point([k[0]-bbox[0],k[1]-bbox[1]])
    print(cropped)
    cropped.save("crop" + str(ii)+ ".jpg")
    ii+=1

img.save("test3.jpg")
# for i in range(len(cropped)):
#     cropped[i].save("crop" + str(i)+ ".jpg")
"""
# print(coco[1])
# for i in coco:
    # print(coco.l)