import cv2
import os
import numpy as np
from deephar.utils import *
import pickle
import json
import random

class MERL5Action(object):
    """Implementation of the MERL dataset for action recignition.
    """

    def __init__(self, anno_path,dataconf=None,poselayout=pa16j2d,clip_size=4):

        self.anno_path = anno_path
        self.clip_size = clip_size

        self.dataconf = dataconf
        self.poselayout = poselayout
        # self.videos_dict,self.bbox_dict,self.urls_dict = self.load_data()
        self.videos_dict,self.videos_dict_num,self.action_dict,self.bbox_dict = self.load_data()
        # print(self.videos_dict_num)
        # print(min(self.videos_dict_num.values()))
        # print(set(self.videos_dict_num.values()))
        self.name_videos = list(self.videos_dict.keys())
        # print(self.name_videos)
        random.shuffle(self.name_videos)
        # self.classes = ["ReachToShelf","RetractFromShelf"]
        self.classes = {"ApproachtoShelf":0,"GlanceatShelf":1,"LookatShelf":2}


    def load_data(self):
        f = open(self.anno_path, 'rb')
        # datas = pickle.load(f)
        datas = json.load(f)
        f.close()
        videos_dict = {}
        videos_dict_num = {}
        action_dict = {}
        bbox_dict = {}
        urls_dict = {}
        # print(datas)
        for i in datas:

            url = i["image"]['url']
            # height = i["image"]['height']
            # width = i["image"]['width']
            # image_name = i["image"]['file_name']
            action = i['action']
            # urls_dict[image_name] = url
            bbox = i['person_bbox']
            bbox2 = []
            video_name = url.split("/")[-2]

            try:
                for b in bbox:
                    bbox_ = [0, 0, 0, 0]
                    bbox_[0] = b[0]
                    bbox_[1] = b[1]
                    bbox_[2] = b[2]
                    bbox_[3] = b[3]
                    bbox2.append(bbox_)
            except:
                print(bbox)
                print(video_name)
            # bbox2.append(bbox)
            bbox_dict[url] = bbox2


            if video_name in videos_dict.keys():
                videos_dict[video_name].append(url)
                videos_dict_num[video_name] += 1
            else:
                videos_dict[video_name] = [url]
                videos_dict_num[video_name] = 1
                action_dict[video_name] = action
        return videos_dict,videos_dict_num,action_dict,bbox_dict

    def load_image(self, url):
        try:
            imgt = cv2.imread(url)/255.0
            bbox = self.bbox_dict[url]
            img = None
            if len(bbox) == 0:
                img = cv2.resize(imgt, (256, 256))
            else:
                bbox = bbox[0]
                try:  # crop image with bbox
                    img = imgt[int(bbox[1]):int(round(bbox[3])), \
                          int(bbox[0]):int(round(bbox[2]))]  # crop image
                except:
                    print(bbox)
                    print("error   : img  ", url)

            img = cv2.resize(img, (256, 256))
            return img
        except:
            warning('Error loading sample key: %d' % (url))
            raise

        # return imgt
    def load_video(self,video_name,num_frame):
        video = []
        if self.videos_dict_num[video_name]<num_frame:
            a = [i for i in range(0,self.videos_dict_num[video_name])]
            while len(a)<num_frame:
                b = random.randrange(self.videos_dict_num[video_name])
                a.append(b)
            a.sort()
            for i in a:
                url = self.videos_dict[video_name][i]
                image = self.load_image(url)
                video.append(image)
        else:
            a = [i for i in range(0, self.videos_dict_num[video_name])]
            while len(a) > num_frame:
                a.pop(random.randrange(len(a)))
            a.sort()
            # print(a)
            for i in a:
                url = self.videos_dict[video_name][i]
                image = self.load_image(url)
                video.append(image)
        return video

    def get_data(self, key,mode=0):
        output = {}
        video_name = self.name_videos[key]
        if key>=self.get_length()-1:
            random.shuffle(self.name_videos)
        # num_frame = self.video1s_dict[video_name]

        frame = self.load_video(video_name,self.clip_size)
        cls = [0,0,0]
        action = self.action_dict[video_name]
        cls[self.classes[action]] = 1

        output['frame'] = frame
        output['merlaction'] = cls
        return output

    def get_shape(self, dictkey):
        if dictkey == 'frame':
            # return self.dataconf.input_shape
            return (self.clip_size,256,256,3)
        if dictkey == 'merlaction':
            # return (self.poselayout.num_joints, self.poselayout.dim+1)
            return (len(self.classes),)

        raise Exception('Invalid dictkey on get_shape!')

    def get_length(self, mode = 0):
        return len(self.videos_dict)


import matplotlib.pyplot as plt
if __name__ == "__main__":
    # anno_path = "/home/pminhtamnb/123.json"
    anno_path = "/mnt/hdd10tb/Users/andang/actions/train_2.json"

    merl = MERL5Action(anno_path,clip_size=8)
    # frame = merl.get_data(500)["frame"]
    # merlaction = merl.get_data(500)["merlaction"]
    # print(len(frame))
    # print(merlaction)
    print(merl.get_length())
    # for i in range(10):
    #     print(merl.get_data(i)["merlaction"])
