# 2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning

Clone from https://github.com/dluvizon/deephar <br/> Add some features

## Add dalaloader, train code with merl dataset and coco
# Coco dataset
For pose estimation. Use pycocotools to get image and label pose Image
had been crop and resize to size (256,256).  <br/>
 Pose have 16 key points,


# Merl dataset
## 1. Merl for pose estimation
Data from pkl with each element have form :
```
{   'img_paths': '17_2_crop_1077_1108_RetractFromShelf-0004.jpg', 
    'img_width': 920,
    'img_height': 680, 
    'image_id': 1,
    'bbox': [602.7314290727887, 324.3657157897949, 218.69714235578266, 119.07430627005441],
    'num_keypoints': 24,
    'keypoints': [[0.0, 0.0, 2], [747.5, 366.0285714285714, 1], [0.0, 0.0, 2], [741.4201450892857, 385.51311383928567, 1],\
               [0.0,0.0, 2], [704.9828591482981, 418.7314265659877, 1], [782.9857142857143, 393.3,0],\
                [632.4342917306083, 353.47713884626114, 1], [0.0, 0.0, 2], [690.2628631591797, 350.8485674176897, 1],\
                [764.5200060163226, 340.00571027483255, 1], [0.0, 0.0,2], [0.0, 0.0, 2], [0.0, 0.0, 2], [0.0, 0.0, 2],\
               [615.348577444894, 386.66285313197545, 1], [0.0, 0.0, 2]]}
```         
With keypoint is list of point , each point is array with form \[
x,y,confident_score \]
## 2. Merl for action recognition
Combine pose and visual feature for action recognition. Data load from
json file with bbox of each frame.

# Run code
Train coco pose estimation
```
 CUDA_VISIBLE_DEVICES=1 python exp/coco/train_coco_singleperson.py
--batch-size 16 --epochs 10
```

Train merl pose estimation

```
 CUDA_VISIBLE_DEVICES=1 python exp/coco/train_merl_singleperson.py
--batch-size 16 --epochs 10
```

Train merl action recognition
```
 CUDA_VISIBLE_DEVICES=1 python exp/coco/train_merl_video.py
--batch-size 16 --epochs 10
```

# Model 
## action
### File **reception.py**

Pose estimation model


- input is list of image, shape = (height,width,channel)
- ouput predict pose estimation

### file **action.py** have : 
- input is list of image, shape =
(num_frames,height,width,channel)
- ouput predict action in onehot encode 
 
### File **action_2D.py** like file **action.py** , delete some code, just run for 2D pose

### File **action_pose.py** model predict from output of pose model
- input 
  - y : pose corrdinate follow time distributed shape = (1, num_frames,
  - num_joints, 2 (x,y - num coordinates))
  - p : probability visible each point shape = (1, num_frames,
    numjoints, 1) 
  - hs : heat map shape = (1, num_frames, 32, 32, num_joints) 
  - xb1: visual feature output from Stem model shape = (1, num_frames,
    32, 32, 576)
- output predict action
## Citing

Please cite our paper if this software (or any part of it) or weights are
useful for you.
```
@InProceedings{Luvizon_2018_CVPR,
  author = {Luvizon, Diogo C. and Picard, David and Tabia, Hedi},
  title = {2D/3D Pose Estimation and Action Recognition Using Multitask Deep Learning},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2018}
}
```

## License

MIT License

