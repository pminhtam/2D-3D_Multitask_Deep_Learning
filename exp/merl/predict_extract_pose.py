import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())
from keras.models import Model
from keras.layers import concatenate
from deephar.config import pennaction_dataconf
from deephar.config import mpii_sp_dataconf

from deephar.models import reception

from deephar.utils import *
import json
import cv2
import numpy as np
from tqdm import tqdm


"""Architecture configuration."""
num_blocks = 8
input_shape = mpii_sp_dataconf.input_shape
num_joints = 16

"""Build pose and action models."""
model = reception.build(input_shape, num_joints, dim=2,
                           num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5),
                           concat_pose_confidence=False)
outputs = []
for b in range(int(len(model.outputs) / 2)):
    outputs.append(concatenate([model.outputs[2*b], model.outputs[2*b + 1]],
        name='blk%d' % (b + 1)))
model = Model(model.input, outputs, name=model.name)

weights_path = "weights_merl_061.h5"
model.load_weights(weights_path)

printcn(OKBLUE,"Load model done")

val_anno_path = "/mnt/hdd10tb/Users/andang/actions/train_2.json"
f = open(val_anno_path, 'rb')
datas = json.load(f)
f.close()
def read_img(url,bbox):
    try:
        imgt = cv2.imread(url) / 255.0
        img = None
        if len(bbox) == 0:
            img = cv2.resize(imgt, (256, 256))
        else:
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
datas_keypoint = list()

with open("train_2_keypoint.json", 'w') as g:
# with open("zzz.json", 'w') as g:
    for i in tqdm(range(len(datas))):
        try:
            data = datas[i]
            json_file = {}
            json_file["image"] = {}
            json_file["image"]["url"] = data["image"]["url"]
            # json_file["image"]["height"] = data["image"]["height"]
            # json_file["image"]["width"] = data["image"]["width"]
            json_file["image"]["file_name"] = data["image"]["file_name"]
            json_file["person_bbox"] = data["person_bbox"]
            img = read_img(data["image"]["url"],data["person_bbox"])
            pred = model.predict(np.array([img]))
            key_point = pred[0].tolist()
            # print(key_point[0])
            json_file["keypoints"] = key_point
            json_file["id"] = data["id"]
            json_file["action"] = data["action"]
            datas_keypoint.append(json_file)
        except:
            printcn(OKBLUE,"error load " + datas[i]['image']['url'])
            continue
        
    json.dump(datas_keypoint, g)
    # print(datas_keypoint)
