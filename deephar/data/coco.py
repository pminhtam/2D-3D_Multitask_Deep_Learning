import cv2
import os
import numpy as np
from pycocotools.coco import COCO
from deephar.utils import *
import random

class COCOSinglePerson(object):
    """Implementation of the COCO dataset for pose estimation.
    """

    def __init__(self, dataset_path, anno_path,dataconf=None,poselayout=pa16j2d):

        self.dataset_path = dataset_path
        self.anno_path = anno_path
        self.coco = COCO(anno_path)
        self.list_image_name = self.load_list_image_name()
        self.dataconf = dataconf
        self.poselayout = poselayout

    def load_list_image_name(self):
        return self.coco.getImgIds()
    def load_annotations(self, annIdx):
        try:
            ann = self.coco.loadAnns(annIdx)[0]
            bbox = ann['bbox']
            # bbox = (bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1])
            key_point = np.array(ann['keypoints']).reshape(17, 3)
            key_point_2 = np.zeros((17,3))
            for i in range(len(key_point)):
                if key_point[i][2]!=0:  # convert keypoint scale [0,1]
                    key_point_2[i][0] = float((key_point[i][0] - bbox[0])/bbox[2])
                    key_point_2[i][1] = ((key_point[i][1] - bbox[1])/bbox[3])
                    key_point_2[i][2] = 1

            return key_point_2,bbox
        except:
            warning('Error loading the MPII dataset!')
            raise

    def load_image(self, imgIdx):
        try:
            image_name = "0"*(12-len(str(imgIdx)))+str(imgIdx) + ".jpg"
            # print(image_name)
            imgt = cv2.imread(os.path.join(self.dataset_path, image_name))/255.0
            return imgt
        except:
            warning('Error loading sample key: %d' % (imgIdx))
            raise

    def get_data(self, key,mode=0):
        output = {}

        imgIdx = self.list_image_name[key]
        img = self.load_image(imgIdx)
        annIds = self.coco.getAnnIds(imgIdx)
        key_point = np.zeros((17,3))
        if len(annIds)==0:  # if haven't anno , pass image , keypoint = array of zero
            pass
        else:               # if have at least 1 pose, choose random pose and bbox
            annIdx = random.choice(annIds)
            key_point, bbox = self.load_annotations(annIdx)
            try:        # crop image with bbox
                img = img[int(bbox[1]):int(round(bbox[1])) + int(round(bbox[3])),\
                      int(bbox[0]):int(round(bbox[0])) +int(round(bbox[2]))]          # crop image
            except:
                print(bbox)
                print("error   : imgIdx  ",imgIdx,"     annIdx   ", annIdx)
        try:
            img = cv2.resize(img, (256, 256))
        except:
            print("error resize  : ",img.shape)
            print("error  resize : imgIdx  ", imgIdx)
        output['image'] = img
        output['pose'] = key_point[:-1]

        return output

    def get_shape(self, dictkey):
        if dictkey == 'image':
            # return self.dataconf.input_shape
            return (256,256,3)
        if dictkey == 'pose':
            # return (self.poselayout.num_joints, self.poselayout.dim+1)
            return (16,3)

        raise Exception('Invalid dictkey on get_shape!')

    def get_length(self, mode):
        return len(self.list_image_name)

import matplotlib.pyplot as plt
if __name__ == "__main__":
    coco = COCOSinglePerson("/mnt/hdd10tb/Users/tam/COCO/val2017","/mnt/hdd10tb/Users/tam/COCO/annotations/person_keypoints_val2017.json")
    data = coco.get_data(11)
    img = data["image"]
    pose = data["pose"]
    print(pose.shape)
    # print(img)
    # plt.imshow(img)
    # for zz in pose:
    #     print(zz)
    #     plt.scatter(zz[0] * 256, zz[1] * 256)
    # plt.savefig("coco_data_5.jpg")