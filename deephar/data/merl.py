import cv2
import os
import numpy as np
from deephar.utils import *
import pickle

class MERLSinglePerson(object):
    """Implementation of the MPII dataset for single person.
    """

    def __init__(self, dataset_path, anno_path,dataconf=None,poselayout=pa16j2d):

        self.dataset_path = dataset_path
        self.anno_path = anno_path
        self.datas = self.load_data()
        self.dataconf = dataconf
        self.poselayout = poselayout

    def load_data(self):
        f = open(self.anno_path, 'rb')
        datas = pickle.load(f)
        f.close()
        return datas

    def load_image(self, image_name):
        try:
            # print(image_name)
            # imgt = Image.open(os.path.join(self.dataset_path, image_name))
            imgt = cv2.imread(os.path.join(self.dataset_path, image_name))/255.0
            return imgt
        except:
            warning('Error loading sample key: %d' % (image_name))
            raise

        # return imgt

    def get_data(self, key,mode=0):
        output = {}

        data = self.datas[key]
        img = self.load_image(data['img_paths'])

        key_point = data['keypoints']
        bbox = data['bbox']
        key_point_2 = np.zeros((17, 3))


        try:    # crop image with bbox
            img = img[int(bbox[1]):int(round(bbox[1])) + int(round(bbox[3])),\
                  int(bbox[0]):int(round(bbox[0])) + int(round(bbox[2]))]  # crop image
        except:
            print(bbox)
            print("error   : img  ",img)

        height,width,_ = img.shape
        try:
            img = cv2.resize(img, (256, 256))
        except:
            print("error resize  : ",img.shape)
            print("erroe resize  :bbox   :  ",bbox)
            print("error  resize : img  ", img)


        for i in range(len(key_point)):     # convert keypoint scale [0,1]
            if key_point[i][2] == 1:
                key_point_2[i][0] = float((key_point[i][0] - bbox[0]) / 256.0)* (256/width)
                key_point_2[i][1] = float((key_point[i][1] - bbox[1]) / 256.0)* (256/height)
                key_point_2[i][2] = 1
            else:
                key_point_2[i][2] = 0
        output['image'] = img
        output['pose'] = key_point_2
        return output

    def get_shape(self, dictkey):
        if dictkey == 'image':
            # return self.dataconf.input_shape
            return (256,256,3)
        if dictkey == 'pose':
            # return (self.poselayout.num_joints, self.poselayout.dim+1)
            return (17,3)

        raise Exception('Invalid dictkey on get_shape!')

    def get_length(self, mode):
        return len(self.datas)


import matplotlib.pyplot as plt
if __name__ == "__main__":
    anno_path = "/home/pminhtamnb/proj4/7-kpts/merl4000_4300.pkl"
    dataset_path = "/mnt/hdd10tb/Users/duong/MERL"
    merl = MERLSinglePerson(dataset_path,anno_path)
    img = merl.get_data(5)["image"]
    pose = merl.get_data(5)["pose"]
    print(img)
    plt.imshow(img)
    for zz in pose:
        print(zz)
        plt.scatter(zz[0] * 256, zz[1] * 256)
    plt.savefig("merl_data_2.jpg")