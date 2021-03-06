import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

from deephar.config import mpii_sp_dataconf
from deephar.data import MERLSinglePerson
from deephar.data import BatchLoader
from deephar.models import reception
from deephar.losses import pose_regression_loss
from keras.optimizers import RMSprop
from deephar.utils import *
from keras.callbacks import LearningRateScheduler
from deephar.callbacks import SaveModel

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
import argparse


sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

annothelper.check_mpii_dataset()

parser = argparse.ArgumentParser()

parser.add_argument("--num-block",type=int,default=8,help="number of block")
parser.add_argument("--batch-size",type=int,default=16,help="batch size")
parser.add_argument("--epochs",type=int,default=100,help="epochs")
parser.add_argument("--num-joints",type=int,default=16,help="num joint")
parser.add_argument("--log-dir",type=str,default=None,help="log dir")
parser.add_argument("--anno-path",type=str,default="/home/duong/lightweight-human-pose-estimation/prepare_train_merl.pkl",\
                    help="annotation path")
parser.add_argument("--image-path",type=str,default="/mnt/hdd10tb/Users/duong/MERL",help="image path")
parser.add_argument("--weights-path",type=str,default=None,help="weights path")
args = parser.parse_args()

weights_file = 'weights_merl_{epoch:03d}.h5'
"""configuration."""
logdir = args.log_dir
num_blocks = args.num_block
batch_size = args.batch_size
epochs = args.epochs

input_shape = mpii_sp_dataconf.input_shape
num_joints = args.num_joints
# anno_path = "/home/pminhtamnb/proj4/7-kpts/merl4000_4300.pkl"
anno_path = args.anno_path
image_path = args.image_path

if logdir:
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log_merl.txt', 'w')

model = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5))

# weights_path = 'weights_merl_021.h5'

weights_path = args.weights_path
if weights_path:
    printcn(WARNING, 'load weights from %s' % (weights_path))
    model.load_weights(weights_path)

"""Load the dataset."""
merl = MERLSinglePerson(image_path,anno_path,dataconf=mpii_sp_dataconf)

data_tr = BatchLoader(merl, ['image'], ['pose'], TRAIN_MODE,
        batch_size=batch_size, num_predictions=num_blocks, shuffle=True)


# """
loss = pose_regression_loss('l1l2bincross', 0.01)
model.compile(loss=loss, optimizer=RMSprop())
# model.summary()

def lr_scheduler(epoch, lr):

    if epoch in [300, 350] or epoch in [50,60]:
        newlr = 0.99*lr
        printcn(WARNING, 'lr_scheduler: lr %g -> %g @ %d' % (lr, newlr, epoch))
    else:
        newlr = lr
        printcn(OKBLUE, 'lr_scheduler: lr %g @ %d' % (newlr, epoch))

    return newlr

callbacks = []
callbacks.append(SaveModel(weights_file,save_after_num_epoch=20))
callbacks.append(LearningRateScheduler(lr_scheduler))

steps_per_epoch = merl.get_length(TRAIN_MODE) // batch_size

model.fit_generator(data_tr,
        # steps_per_epoch=steps_per_epoch,
        epochs=1000,
        callbacks=callbacks,
        workers=8,
        initial_epoch=0)

# """