import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from keras.utils.data_utils import get_file

from deephar.config import pennaction_dataconf

from deephar.data import MERLAction
from deephar.data import MERL5Action
from deephar.data import BatchLoader

from deephar.models import reception
from deephar.models import action_2D as action
from deephar.utils import *
from deephar.callbacks import SaveModel
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from keras.optimizers import SGD

import argparse

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))


sys.path.append(os.path.join(os.getcwd(), 'datasets'))


parser = argparse.ArgumentParser()

parser.add_argument("--num-frames",type=int,default=8,help="number frame per video")
parser.add_argument("--num-block",type=int,default=8,help="number of block")
parser.add_argument("--batch-size",type=int,default=2,help="batch size",metavar='b')
parser.add_argument("--epochs",type=int,default=1000,help="epochs",metavar='e')
parser.add_argument("--num-joints",type=int,default=16,help="num joint",metavar='j')
parser.add_argument("--learning-rate",type=float,default=0.001,help="num joint",metavar='lr')
parser.add_argument("--num-actions",type=int,default=3,help="num action")
parser.add_argument("--anno-path",type=str,default="/mnt/hdd10tb/Users/andang/actions/train_2.json",\
                    help="annotation path")
parser.add_argument("--val-anno-path",type=str,default="/mnt/hdd10tb/Users/andang/actions/test_2.json",\
                    help="validation annotation path")
parser.add_argument("--weights-pose-path",type=str,default="weights_merl_061.h5",help="weights pose path")
parser.add_argument("--weights-action-path",type=str,default=None,help="weights action path")
args = parser.parse_args()


num_frames = args.num_frames
use_bbox = False
# use_bbox = False
num_blocks = args.num_block
batch_size = args.batch_size
epochs = args.epochs
input_shape = pennaction_dataconf.input_shape
num_joints = args.num_joints
num_actions = args.num_actions
lr = args.learning_rate
momentum = 0.95
loss_weights=None
anno_path = args.anno_path
val_anno_path = args.val_anno_path

model_pe = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5))
weights_path = args.weights_pose_path
model_pe.load_weights(weights_path)

printcn(HEADER,"Load model reception done")

model = action.build_merge_model(model_pe, num_actions, input_shape,
        num_frames, num_joints, num_blocks, pose_dim=2, pose_net_version='v1',
        full_trainable=True)

weights_action_path = args.weights_action_path
if weights_action_path !=None:
    model.load_weights(weights_action_path)

printcn(OKBLUE,"Load model done")
# weights_path = "weights_merlaction_1_003.h5"
# model.load_weights(weights_path)
# model.summary()
# """

# anno_path = "/mnt/hdd10tb/Users/andang/actions/train_2.json"
# videos_dict_path =  "/home/son/lightweight-human-pose-estimation/settings/3/train_merl.json"
# videos_dict_path =  "/mnt/hdd10tb/Users/pminhtamnb/deephar/settings/setting_5/train.txt"
merl_seq = MERL5Action(anno_path,pennaction_dataconf,
        poselayout=pa16j2d, clip_size=num_frames)

merl_te = BatchLoader(merl_seq, ['frame'], ['merlaction'], TRAIN_MODE,
        batch_size=batch_size, shuffle=False,num_predictions=11)

printcn(WARNING,"Data train " + str(merl_seq.get_length()))
printcn(OKGREEN,"Load data train done")

# val_anno_path = "/mnt/hdd10tb/Users/andang/actions/test_2.json"
val_merl_seq = MERL5Action(val_anno_path,pennaction_dataconf,
        poselayout=pa16j2d, clip_size=num_frames)
val_merl_te = BatchLoader(val_merl_seq, ['frame'], ['merlaction'], TRAIN_MODE,
        batch_size=batch_size, shuffle=False,num_predictions=11)

printcn(WARNING,"Data test " + str(val_merl_seq.get_length()))
printcn(OKGREEN,"Load data test done")

callbacks = []
weights_file = "/mnt/hdd10tb/Users/pminhtamnb/deephar/weights_merlaction_new_4_{epoch:03d}-{val_m_acc:.4f}.h5"

# callbacks.append(SaveModel(weights_file,save_after_num_epoch=10))
callbacks.append(ModelCheckpoint(weights_file, monitor='val_m_acc', verbose=1, save_best_only=True, mode='max',period=5))
callbacks.append(EarlyStopping(monitor='val_m_acc', min_delta=0, patience=10, verbose=1, mode='auto'))
# callbacks.append(TensorBoard(log_dir='./log/', histogram_freq=0,write_graph=False, write_images=False,update_freq = 500))
callbacks.append(TensorBoard(log_dir='./log/', histogram_freq=0,write_graph=False, write_images=False))




model.compile(loss='categorical_crossentropy',
                optimizer=SGD(lr=lr, momentum=momentum, nesterov=True),
                metrics=['acc'], loss_weights=loss_weights)

printcn(OKBLUE,"model compile done")

model.fit_generator(merl_te,
        epochs=epochs,
        callbacks=callbacks,
        workers=4,
        validation_data=val_merl_te,
        shuffle=True,
        initial_epoch=0)
# for layer in model.layers:
#         layer.trainable = True
model.save_weights('weights_merlaction_new_done.h5')
