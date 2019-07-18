import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())



from deephar.config import pennaction_dataconf

from deephar.data import PennAction
from deephar.data import BatchLoader

from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from penn_tools import eval_singleclip_generator
from penn_tools import eval_multiclip_dataset

sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

annothelper.check_pennaction_dataset()


num_frames = 16
use_bbox = False
# use_bbox = False
num_blocks = 4
batch_size = 2
input_shape = pennaction_dataconf.input_shape
num_joints = 16
num_actions = 15





"""Load PennAction dataset."""
# """
penn_seq = PennAction('datasets/PennAction', pennaction_dataconf,
        poselayout=pa16j2d, topology='sequences', use_gt_bbox=use_bbox,
        clip_size=num_frames)

penn_te = BatchLoader(penn_seq, ['frame'], ['pennaction'], TRAIN_MODE,
        batch_size=1, shuffle=False)



data = penn_te[1]
print(data)
