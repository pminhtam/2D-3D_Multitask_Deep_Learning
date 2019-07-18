import pickle
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# f = open("/home/duong/lightweight-human-pose-estimation/prepare_train_merl.pkl",'rb')
anno_path= "/home/duong/lightweight-human-pose-estimation/prepare_train_merl.pkl"
dataset_path = "/mnt/hdd10tb/Users/duong/MERL"


"""
images = []
bbox_dict = {}
datas = pickle.load(f)
f.close()

for i in datas:
    images.append(i["img_paths"])
    bbox_dict[i['img_paths']] = i['bbox']
classes = ["ReachToShelf","RetractFromShelf"]


name_videos = {}
for i in images:
    name_video = i.split("-")[0]
    if name_video in name_videos.keys():
        name_videos[name_video] +=1
    else:
        name_videos[name_video] = 1

# print(name_videos)

# print(list(name_videos.keys())[2])
name = list(name_videos.keys())[10]+"-0001.jpg"
print(name)
print(bbox_dict[name])
print(min(name_videos.values()))

video = []
for i in range(1,4):
    img = cv2.imread(os.path.join(dataset_path,'17_2_crop_1077_1108_RetractFromShelf-000'+str(i)+'.jpg'))
    video.append(img)

print(video)
print(np.array(video).shape)


# print(len(name_videos))   # 396
# print(len(images))
"""


def load_data():
    f = open(anno_path, 'rb')
    datas = pickle.load(f)
    f.close()
    name_videos = {}
    bbox_dict = {}
    for i in datas:
        bbox_dict[i['img_paths']] = i['bbox']
        name_video = i['img_paths'].split("-")[0]
        if name_video in name_videos.keys():
            name_videos[name_video] += 1
        else:
            name_videos[name_video] = 1
    return name_videos, bbox_dict

name_videos, bbox_dict = load_data()

print(len(name_videos))
# print(bbox_dict)