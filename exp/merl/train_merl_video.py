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
import keras

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))


sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

# annothelper.check_pennaction_dataset()



num_frames = 8
use_bbox = False
# use_bbox = False
num_blocks = 8
batch_size = 2
input_shape = pennaction_dataconf.input_shape
num_joints = 16
num_actions = 3


model_pe = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5))
weights_path = "weights_merl_061.h5"
model_pe.load_weights(weights_path)
printcn(HEADER,"Load model reception done")

model = action.build_merge_model(model_pe, num_actions, input_shape,
        num_frames, num_joints, num_blocks, pose_dim=2, pose_net_version='v1',
        full_trainable=False)

printcn(OKBLUE,"Load model done")
# weights_path = "weights_merlaction_1_003.h5"
# model.load_weights(weights_path)
# model.summary()
# """

anno_path = "/mnt/hdd10tb/Users/andang/actions/train_2.json"
# videos_dict_path =  "/home/son/lightweight-human-pose-estimation/settings/3/train_merl.json"
# videos_dict_path =  "/mnt/hdd10tb/Users/pminhtamnb/deephar/settings/setting_5/train.txt"
merl_seq = MERL5Action(anno_path,pennaction_dataconf,
        poselayout=pa16j2d, clip_size=num_frames)

merl_te = BatchLoader(merl_seq, ['frame'], ['merlaction'], TRAIN_MODE,
        batch_size=batch_size, shuffle=False,num_predictions=11)

printcn(WARNING,"Data train " + str(merl_seq.get_length()))
printcn(OKGREEN,"Load data train done")

val_anno_path = "/mnt/hdd10tb/Users/andang/actions/test_2.json"
val_merl_seq = MERL5Action(val_anno_path,pennaction_dataconf,
        poselayout=pa16j2d, clip_size=num_frames)
val_merl_te = BatchLoader(merl_seq, ['frame'], ['merlaction'], TRAIN_MODE,
        batch_size=batch_size, shuffle=False,num_predictions=11)

printcn(WARNING,"Data test " + str(val_merl_seq.get_length()))
printcn(OKGREEN,"Load data test done")

callbacks = []
weights_file = 'weights_merlaction_new_{epoch:03d}.h5'

callbacks.append(SaveModel(weights_file,save_after_num_epoch=10))
callbacks.append(keras.callbacks.TensorBoard(log_dir='./log/', histogram_freq=0,write_graph=False, write_images=False))

from keras.optimizers import SGD
lr = 0.001
momentum = 0.95
loss_weights=None
epochs = 500
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
# """

model.save_weights(weights_file.format(epoch =epochs))