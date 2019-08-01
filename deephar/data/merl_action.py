import cv2
import os
import numpy as np
from deephar.utils import *
import pickle

class MERLAction(object):
    """Implementation of the MERL dataset for action recignition.
    """

    def __init__(self, dataset_path, anno_path,dataconf=None,poselayout=pa16j2d,clip_size=3):

        self.dataset_path = dataset_path
        self.anno_path = anno_path

        self.dataconf = dataconf
        self.poselayout = poselayout
        self.videos_dict,self.bbox_dict = self.load_data()
        self.name_videos = list(self.videos_dict.keys())
        self.classes = ["ReachToShelf","RetractFromShelf"]
        self.clip_size = clip_size

    def load_data(self):
        f = open(self.anno_path, 'rb')
        datas = pickle.load(f)
        f.close()
        videos_dict = {}
        bbox_dict = {}
        for i in datas:
            bbox_dict[i['img_paths']] = i['bbox']
            image_name = i['img_paths']
            name_video = image_name.split("-")[0]
            if name_video in videos_dict.keys():
                videos_dict[name_video].append(image_name)
            else:
                videos_dict[name_video] = [image_name]
        return  videos_dict,bbox_dict

    def load_image(self, image_name):
        try:
            imgt = cv2.imread(os.path.join(self.dataset_path, image_name))/255.0
            bbox = self.bbox_dict[image_name]
            img = np.zeros((int(round(bbox[3])),int(round(bbox[2]))))
            try:  # crop image with bbox
                img = imgt[int(bbox[1]):int(round(bbox[1])) + int(round(bbox[3])), \
                      int(bbox[0]):int(round(bbox[0])) + int(round(bbox[2]))]  # crop image
            except:
                print(bbox)
                print("error   : img  ", image_name)

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
            return (3,256,256,3)
        if dictkey == 'merlaction':
            # return (self.poselayout.num_joints, self.poselayout.dim+1)
            return (2,)

        raise Exception('Invalid dictkey on get_shape!')

    def get_length(self, mode):
        return len(self.videos_dict)


import matplotlib.pyplot as plt
if __name__ == "__main__":
    anno_path = "/home/pminhtamnb/proj4/7-kpts/merl4000_4300.pkl"
    dataset_path = "/mnt/hdd10tb/Users/duong/MERL"
    merl = MERLAction(dataset_path,anno_path)
    frame = merl.get_data(5)["frame"]
    merlaction = merl.get_data(5)["merlaction"]
    print(len(frame))
    print(merlaction)
