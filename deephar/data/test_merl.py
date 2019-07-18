import pickle
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# f = open("/home/pminhtamnb/proj4/7-kpts/merl4000_4300.pkl",'rb')
f = open("/home/duong/lightweight-human-pose-estimation/prepare_train_merl.pkl",'rb')
dataset_path = "/mnt/hdd10tb/Users/duong/MERL"

datas = pickle.load(f)
f.close()
# print(data[0])
# print(data)
data = datas[1]
# print(data)
# {'img_paths': '17_2_crop_1077_1108_RetractFromShelf-0004.jpg', 'img_width': 920,
#  'img_height': 680, 'objpos': [712.0800002506801, 383.9028689248221], 'image_id': 1,\
#  'bbox': [602.7314290727887, 324.3657157897949, 218.69714235578266, 119.07430627005441], 'segment_area': 25606,\
#  'scale_provided': 0.3235714844294957, 'num_keypoints': 24, 'segmentations': [],\
#  'keypoints': [[0.0, 0.0, 2], [747.5, 366.0285714285714, 1], [0.0, 0.0, 2], [741.4201450892857, 385.51311383928567, 1],\
#                [0.0,0.0, 2], [704.9828591482981, 418.7314265659877, 1], [782.9857142857143, 393.3,0],\
#                 [632.4342917306083, 353.47713884626114, 1], [0.0, 0.0, 2], [690.2628631591797, 350.8485674176897, 1],\
#                 [764.5200060163226, 340.00571027483255, 1], [0.0, 0.0,2], [0.0, 0.0, 2], [0.0, 0.0, 2], [0.0, 0.0, 2],\
#                [615.348577444894, 386.66285313197545, 1], [0.0, 0.0, 2]], 'processed_other_annotations': []}


# """
image_name = data['img_paths']
bbox = data['bbox']

img = cv2.imread(os.path.join(dataset_path, image_name))/255.0

img = img[int(bbox[1]):int(round(bbox[1])) + int(round(bbox[3])),\
                  int(bbox[0]):int(round(bbox[0])) + int(round(bbox[2]))]


height,width,_ = img.shape

img = cv2.resize(img, (256, 256))

print(height, "  ",width)
key_point = data['keypoints']

key_point_2 = np.zeros((17, 3))

for i in range(len(key_point)):     # convert keypoint scale [0,1]
    if key_point[i][2] == 1:
        key_point_2[i][0] = float((key_point[i][0] - bbox[0]) / 256.0) * (256/width)
        key_point_2[i][1] = float((key_point[i][1] - bbox[1]) / 256.0) * (256/height)
        key_point_2[i][2] = 1
    else:
        key_point_2[i][2] = 0


plt.imshow(img)
for zz in key_point_2:
    print(zz)
    plt.scatter(zz[0]*256, zz[1]*256)
plt.savefig("merl_data_1.jpg")
# """