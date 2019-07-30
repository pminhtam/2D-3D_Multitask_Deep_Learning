import cv2
import os
import numpy as np
from deephar.utils import *
import pickle
import json
import random

class MERL5Action(object):
    """Implementation of the MPII dataset for single person.
    """

    def __init__(self,videos_dict_path, anno_path,dataconf=None,poselayout=pa16j2d,clip_size=4):

        self.anno_path = anno_path
        self.videos_dict_path = videos_dict_path
        self.clip_size = clip_size

        self.dataconf = dataconf
        self.poselayout = poselayout
        # self.videos_dict,self.bbox_dict,self.urls_dict = self.load_data()
        __,self.bbox_dict,self.urls_dict = self.load_data()
        self.videos_dict = self.load_videos_dict()

        self.name_videos = list(self.videos_dict.keys())
        random.shuffle(self.name_videos)
        self.classes = ["ReachToShelf","RetractFromShelf"]

    def load_videos_dict(self):
        f = open(self.videos_dict_path, 'r')
        datas = []
        for i in f:
            datas.append(i.strip("\n"))
        f.close()
        videos_dict = {}
        videos_dict_num = {}
        for i in datas:
            # image_name = i
            # video_name = image_name.split("-")[0]
            video_name = i
            """
            if video_name in videos_dict.keys():
                videos_dict[video_name].append(image_name)
                videos_dict_num[video_name]+=1
            else:
                videos_dict[video_name] = [image_name]
                videos_dict_num[video_name] = 1
                
        for i in videos_dict_num.keys():            # if video length have frame less than clip_size. Delete it
            if videos_dict_num[i]<self.clip_size:
                del videos_dict[i]
                """
            if video_name not in videos_dict.keys():
                videos_dict[video_name] = []
                videos_dict_num[video_name] = 0
                for j in range(1,6):
                    videos_dict[video_name].append(video_name+"-000"+str(j)+".jpg")
                    videos_dict_num[video_name] += 1
        return videos_dict

    def load_data(self):
        f = open(self.anno_path, 'rb')
        # datas = pickle.load(f)
        datas = json.load(f)
        f.close()
        videos_dict = {}
        bbox_dict = {}
        urls_dict = {}
        # print(datas)
        for i in datas:

            url = os.path.join("/mnt/hdd10tb/Datasets/MERL3000",i["image"]['url'].split("/")[-1])
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

        return videos_dict, bbox_dict,urls_dict

    def load_image(self, image_name):
        try:
            url = self.urls_dict[image_name]
            imgt = cv2.imread(url)/255.0
            bbox = self.bbox_dict[image_name]

            if len(bbox) == 0:
                img = cv2.resize(imgt, (256, 256))
                return img
            # bbox = random.choice(bbox)
            bbox = bbox[0]
            img = np.zeros((int(round(bbox[3])),int(round(bbox[2]))))
            try:  # crop image with bbox
                img = imgt[int(bbox[1]):int(round(bbox[1])) + int(round(bbox[3])), \
                      int(bbox[0]):int(round(bbox[0])) + int(round(bbox[2]))]  # crop image
            except:
                print(bbox)
                print("error   : img  ", url)

            img = cv2.resize(img, (256, 256))
            return img
        except:
            warning('Error loading sample key: %d' % (image_name))
            raise

        # return imgt
    def load_video(self,video_name,num_frame):
        video = []
        for i in range(0,num_frame):
            if i>=num_frame:
                break
            image_name = self.videos_dict[video_name][i]
            image = self.load_image(image_name)
            video.append(image)
        return video

    def get_data(self, key,mode=0):
        output = {}
        video_name = self.name_videos[key]
        if key>=self.get_length()-1:
            random.shuffle(self.name_videos)
        # num_frame = self.video1s_dict[video_name]

        frame = self.load_video(video_name,self.clip_size)
        cls = [0,0]
        if self.classes[0] in video_name:
            cls[0] = 1
        elif self.classes[1] in video_name:
            cls[1] = 1

        output['frame'] = frame
        output['merlaction'] = cls
        return output

    def get_shape(self, dictkey):
        if dictkey == 'frame':
            # return self.dataconf.input_shape
            return (self.clip_size,256,256,3)
        if dictkey == 'merlaction':
            # return (self.poselayout.num_joints, self.poselayout.dim+1)
            return (2,)

        raise Exception('Invalid dictkey on get_shape!')

    def get_length(self, mode = 0):
        return len(self.videos_dict)


import matplotlib.pyplot as plt
if __name__ == "__main__":
    anno_path = "/home/pminhtamnb/data.json"
    videos_dict_path = "/home/son/lightweight-human-pose-estimation/settings/1/train_merl.json"

    merl = MERL5Action(videos_dict_path,anno_path)
    frame = merl.get_data(5)["frame"]
    merlaction = merl.get_data(5)["merlaction"]
    print(len(frame))
    print(merlaction)
