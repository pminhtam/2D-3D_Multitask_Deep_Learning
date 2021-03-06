import os

import json
import numpy as np
import scipy.io as sio
from PIL import Image

from deephar.data.datasets import get_clip_frame_index
from deephar.utils import *

ACTION_LABELS = None

def load_pennaction_mat_annotation(filename):
    mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    # Respect the order of TEST (0), TRAIN (1). No validation set.
    sequences = [mat['sequences_te'], mat['sequences_tr'], []]

    # sequences gồm sequence_test và sequences_train
    # mỗi sequences con lại gồm nhiều video
    # sequences[0][0]       video số 1
    # sequences[0][0].frames    tập các frame của một video
    # sequences[0][0].frames[0]  một frame của 1 video
    # sequences[0][0].frames[0].pose    gồm các thuộc tính pose
    #                                             [[136.5  135.   102.75 160.5  141.   162.   154.5  114.75 107.25 142.5
    #                                               102.   141.75  93.]
    #                                              [71.25 102.75  99.   148.5  148.5  117.75 122.25 175.5  180.   255.
    #                                              255.75 324.75 321.]]

    # sequences[0][0].frames[0].visible                     visible
    #                                                 [1 1 0 1 0 1 0 1 0 1 1 1 1]

    action_labels = mat['action_labels']
    # Mảng tên các action trong bộ dữ liệu
    #             ['baseball_pitch' 'baseball_swing' 'bench_press' 'bowl' 'clean_and_jerk'
    #              'golf_swing' 'jump_rope' 'jumping_jacks' 'pullup' 'pushup' 'situp'
    #              'squat' 'strum_guitar' 'tennis_forehand' 'tennis_serve']

    joint_labels = mat['joint_labels']
    # Mảng tên các pose theo thứ tự
    #             ['head' 'r_shoulder' 'l_shoulder' 'r_elbow' 'l_elbow' 'r_hand' 'l_hand'
    #              'r_hip' 'l_hip' 'r_knee' 'l_knee' 'r_ankle' 'l_ankle']

    return sequences, action_labels, joint_labels


def serialize_index_sequences(sequences):
    frame_idx = []
    for s in range(len(sequences)):
        for f in range(len(sequences[s].frames)):
            frame_idx.append((s, f))

    return frame_idx


def compute_clip_bbox(bbox_dict, seq_idx, frame_list):
    x1 = y1 = np.inf
    x2 = y2 = -np.inf

    for f in frame_list:
        b = bbox_dict['%d.%d' % (seq_idx, f)]
        x1 = min(x1, b[0])
        y1 = min(y1, b[1])
        x2 = max(x2, b[2])
        y2 = max(y2, b[3])

    return np.array([x1, y1, x2, y2])


class PennAction(object):
    def __init__(self, dataset_path, dataconf, poselayout=pa16j2d,
            topology='sequence', use_gt_bbox=False, remove_outer_joints=True,
            clip_size=16, pose_only=False, output_fullframe=False,
            pred_bboxes_file=None):

        assert topology in ['sequences', 'frames'], \
                'Invalid topology ({})'.format(topology)

        self.dataset_path = dataset_path
        self.dataconf = dataconf
        self.poselayout = poselayout
        self.topology = topology
        self.use_gt_bbox = use_gt_bbox
        self.remove_outer_joints = remove_outer_joints
        self.clip_size = clip_size
        self.pose_only = pose_only
        self.output_fullframe = output_fullframe
        self.load_annotations(os.path.join(dataset_path, 'annotations.mat'))
        if pred_bboxes_file:
            filepath = os.path.join(dataset_path, pred_bboxes_file)
            with open(filepath, 'r') as fid:
                self.pred_bboxes = json.load(fid)
        else:
            self.pred_bboxes = None

    def load_annotations(self, filename):
        try:
            self.sequences, self.action_labels, self.joint_labels = \
                    load_pennaction_mat_annotation(filename)
            self.frame_idx = [serialize_index_sequences(self.sequences[0]),
                    serialize_index_sequences(self.sequences[1]), []]

            global ACTION_LABELS
            ACTION_LABELS = self.action_labels

        except:
            warning('Error loading PennAction dataset!')
            raise

    def get_data(self, key, mode, frame_list=None, bbox=None):
        """Method to load Penn Action samples specified by mode and key,
        do data augmentation and bounding box cropping.
        """
        output = {}

        if mode == TRAIN_MODE:
            dconf = self.dataconf.random_data_generator()
            random_clip = True
        else:
            dconf = self.dataconf.get_fixed_config()
            random_clip = False

        if self.topology == 'sequences':
            seq_idx = key
            seq = self.sequences[mode][seq_idx]
            if frame_list == None:
                frame_list = get_clip_frame_index(len(seq.frames),
                        dconf['subspl'], self.clip_size,
                        random_clip=random_clip)
        else:
            seq_idx, frame_idx = self.frame_idx[mode][key]
            seq = self.sequences[mode][seq_idx]
            frame_list = [frame_idx]

        objframes = seq.frames[frame_list]

        """Load pose annotation"""
        pose, visible = self.get_pose_annot(objframes)
        w, h = (objframes[0].w, objframes[0].h)

        """Compute cropping bounding box, if not given."""
        if bbox is None:

            if self.use_gt_bbox:
                bbox = get_gt_bbox(pose[:, :, 0:2], visible, (w, h),
                        scale=dconf['scale'], logkey=key)

            elif self.pred_bboxes:
                bbox = compute_clip_bbox(
                        self.pred_bboxes[mode], seq_idx, frame_list)

            else:
                bbox = objposwin_to_bbox(np.array([w / 2, h / 2]),
                        (dconf['scale']*max(w, h), dconf['scale']*max(w, h)))

        objpos, winsize = bbox_to_objposwin(bbox)       # objpos tâm của bbox
                                                        # winsize kích thước của bbox : chiều dài và rộng
        if min(winsize) < 32:
            winsize = (32, 32)
        objpos += dconf['scale'] * np.array([dconf['transx'], dconf['transy']])

        """Pre-process data for each frame"""
        if self.pose_only:
            frames = None
        else:
            frames = np.zeros((len(objframes),) + self.dataconf.input_shape)
            if self.output_fullframe:
                fullframes = np.zeros((len(objframes), h, w,
                    self.dataconf.input_shape[-1]))

        for i in range(len(objframes)):
            if self.pose_only:
                imgt = T(None, img_size=(w, h))
            else:
                image = 'frames/%04d/%06d.jpg' % (seq.idx, objframes[i].f)
                # print("PennAction.getdata 175 :  ",image)
                imgt = T(Image.open(os.path.join(self.dataset_path, image)))
                if self.output_fullframe:
                    fullframes[i, :, :, :] = normalize_channels(imgt.asarray(),
                            channel_power=dconf['chpower'])

            imgt.rotate_crop(dconf['angle'], objpos, winsize)
            imgt.resize(self.dataconf.crop_resolution)

            if dconf['hflip'] == 1:
                imgt.horizontal_flip()

            imgt.normalize_affinemap()
            if not self.pose_only:
                frames[i, :, :, :] = normalize_channels(imgt.asarray(),
                        channel_power=dconf['chpower'])

            pose[i, :, 0:2] = transform_2d_points(imgt.afmat, pose[i, :, 0:2],
                    transpose=True)
            if imgt.hflip:
                pose[i, :, :] = pose[i, self.poselayout.map_hflip, :]

        """Set outsider body joints to invalid (-1e9)."""
        pose = np.reshape(pose, (-1, self.poselayout.dim))
        pose[np.isnan(pose)] = -1e9
        v = np.expand_dims(get_visible_joints(pose[:,0:2]), axis=-1)
        pose[(v==0)[:,0],:] = -1e9
        pose = np.reshape(pose, (len(objframes), self.poselayout.num_joints,
            self.poselayout.dim))
        v = np.reshape(v, (len(objframes), self.poselayout.num_joints, 1))

        pose = np.concatenate((pose, v), axis=-1)
        if self.topology != 'sequences':
            pose = np.squeeze(pose, axis=0)
            if not self.pose_only:
                frames = np.squeeze(frames, axis=0)

        action = np.zeros(self.get_shape('pennaction'))
        action[seq.action_id - 1] = 1.

        output['seq_idx'] = seq_idx     # index
        output['frame_list'] = frame_list
        # output['pennaction'] = action
        # print(action)
        # print(action.shape)
        output['pennaction'] = action
        output['ntuaction'] = np.zeros((60,))
        output['pose'] = pose
        output['frame'] = frames
        if self.output_fullframe and not self.pose_only:
            output['fullframe'] = fullframes

        output['bbox'] = bbox

        """Take the last transformation matrix, it should not change"""
        output['afmat'] = imgt.afmat.copy()

        return output

    def get_clip_index(self, key, mode, subsamples=[2]):    # list of frame index . exp [1,3,5,7,9,...]
        assert self.topology == 'sequences', 'Topology not supported'

        seq = self.sequences[mode][key]
        index_list = []
        for sub in subsamples:
            start_frame = 0
            while True:
                last_frame = start_frame + self.clip_size * sub
                if last_frame > len(seq.frames):
                    break
                index_list.append(range(start_frame, last_frame, sub))
                start_frame += int(self.clip_size / 2) + (sub - 1)

        return index_list


    def get_pose_annot(self, frames):
        p = np.nan * np.ones((len(frames), self.poselayout.num_joints,
            self.poselayout.dim))
        v = np.zeros((len(frames), self.poselayout.num_joints))
        for i in range(len(frames)):
            p[i, self.poselayout.map_to_pa13j, 0:2] = frames[i].pose.copy().T
            v[i, self.poselayout.map_to_pa13j] = frames[i].visible.copy()
            p[i, v[i] == 0, :] = np.nan
            p[i, p[i] == 0] = np.nan

        return p, v

    def clip_length(self):
        if self.topology == 'sequences':
            return self.clip_size
        else:
            return None

    def clip_shape(self):
        if self.topology == 'sequences':
            return (self.clip_size,)
        else:
            return ()

    def get_shape(self, dictkey):
        if dictkey == 'frame':
            return self.clip_shape() + self.dataconf.input_shape
        if dictkey == 'pose':
            return self.clip_shape() \
                    + (self.poselayout.num_joints, self.poselayout.dim+1)
        if dictkey == 'pennaction':
            return (len(self.action_labels),)
        if dictkey == 'ntuaction':
            return (60,)
        if dictkey == 'afmat':
            return (3, 3)
        raise Exception('Invalid dictkey ({}) on get_shape!'.format(dictkey))

    def get_length(self, mode): # đưa ra số lượng video trong tập
        if self.topology == 'sequences':
            return len(self.sequences[mode])
        else:
            return len(self.frame_idx[mode])


from deephar.config import pennaction_dataconf
import matplotlib.pyplot as plt
import cv2
if __name__ == "__main__":
    """
    num_frames = 16
    use_bbox = False

    penn_seq = PennAction('datasets/PennAction', pennaction_dataconf,
                          poselayout=pa16j2d, topology='sequences', use_gt_bbox=use_bbox,
                          clip_size=num_frames,output_fullframe=True,pose_only = False)

    data = penn_seq.get_data(1,TEST_MODE)
    # print(data)

    pose = data['pose'][10]
    frame = data['fullframe']+1
    print(pose)
    print("full frame shape   : ",frame.shape)
    # print("frame    ",frame)
    # img = cv2.cvtColor(cv2.imread("datasets/PennAction/frames/0001/0000"+str(frame[0])+".jpg"),cv2.COLOR_RGB2BGR)
    plt.imshow(frame[10])
    x,y,_ = frame[0].shape
    for i in range(16):
        if pose[i][2]>0.01:
            # plt.scatter(pose[i][0]*x,pose[i][1]*y)
            plt.scatter(pose[i][0]*y,pose[i][1]*x)
            # plt.scatter(pose[i][1]*x,pose[i][0]*y)
            # plt.scatter(pose[i][1]*y,pose[i][0]*x)
    plt.savefig("a.png")
    """
    filename = "datasets/PennAction/annotations.mat"
    sequences, action_labels, joint_labels = load_pennaction_mat_annotation(filename)
    mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    # print(sequences)
    # print(sequences[0])
    # print(sequences[0][2].idx)
    # print(sequences[0][1].frames)
    # print(sequences[0][0].frames[0])
    # frames = list(range(0, 0 + 12, 2))
    # print(sequences[0][0].frames[0].pose)
    # print(sequences[0][0].frames[0]._fieldnames) # ['f', 'w', 'h', 'pose', 'visible']
    print(sequences[0][0].frames[5].f)
    # print(sequences[0][0].frames[0].visible)

    # for i in sequences[0][0].frames[0]:
    #     print(i)

    # print(action_labels)
    print(joint_labels)


#   python -m deephar.data.pennaction