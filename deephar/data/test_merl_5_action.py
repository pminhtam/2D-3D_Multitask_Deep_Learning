import pickle
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import json

# f = open("/home/duong/lightweight-human-pose-estimation/prepare_train_merl.pkl",'rb')
anno_path= "/home/pminhtamnb/data.json"


def load_data():
    f = open(anno_path, 'rb')
    # datas = pickle.load(f)
    datas = json.load(f)
    f.close()
    videos_dict = {}
    bbox_dict = {}
    urls_dict = {}
    # print(datas)
    for i in datas:

        url = i["image"]['url']
        height = i["image"]['height']
        width = i["image"]['width']
        image_name = i["image"]['file_name']

        urls_dict[image_name] = url
        bbox = i['person_bbox']
        bbox2 = []
        for b in bbox:
            bbox_ = [.0, .0, .0, .0]
            bbox_[0] = b[0] * width
            bbox_[1] = b[1] * height
            bbox_[2] = b[2] * width
            bbox_[3] = b[3] * width
            bbox2.append(bbox_)
        bbox_dict[image_name] = bbox2

        video_name = image_name.split("-")[0]
        if video_name in videos_dict.keys():
            videos_dict[video_name].append(image_name)
        else:
            videos_dict[video_name] = [image_name]

    return videos_dict, bbox_dict, urls_dict

# videos_dict, bbox_dict,urls_dict = load_data()

# print(len(name_videos))
# print(len(bbox_dict))
# print(bbox_dict)
# print(bbox_dict)
# print(urls_dict.keys())
# print(len(videos_dict.values()))
# print(max(videos_dict.values()))

# f = open(anno_path, 'rb')
# datas = pickle.load(f)
# datas = json.load(f)
# f.close()
# print(datas[1]['image'])
videos_dict_path = "/home/son/lightweight-human-pose-estimation/settings/1/train_merl.json"
def load_videos_dict():
    f = open(videos_dict_path, 'rb')
    # datas = pickle.load(f)
    datas = json.load(f)
    f.close()
    videos_dict = {}
    for i in datas['images']:
        image_name = i['file_name']
        video_name = image_name.split("-")[0]
        if video_name in videos_dict.keys():
            # videos_dict[video_name].append(image_name)
            videos_dict[video_name]+=1
        else:
            # videos_dict[video_name] = [image_name]
            videos_dict[video_name] = 1
    return videos_dict

videos_dict = load_videos_dict()
print(set(videos_dict.values()))