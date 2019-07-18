import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar


from deephar.config import mpii_sp_dataconf

from deephar.data import MpiiSinglePerson
from deephar.data import BatchLoader

from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from mpii_tools import eval_singleperson_pckh

sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

annothelper.check_mpii_dataset()



"""Load the MPII dataset."""
mpii = MpiiSinglePerson('datasets/MPII', dataconf=mpii_sp_dataconf)

"""Pre-load validation samples and generate the eval. callback."""

mpii_val = BatchLoader(mpii, x_dictkeys=['frame'],
        y_dictkeys=['pose'], mode=TRAIN_MODE,
        # batch_size=mpii.get_length(VALID_MODE), num_predictions=1,
        batch_size=1, num_predictions=1,
        shuffle=False)
printcn(OKBLUE, 'Pre-loading MPII validation data...')
# img 256x256
[x_val], [p_val] = mpii_val[10]
print(p_val)


# print(p_val)
# plot data
import matplotlib.pyplot as plt

plt.imshow(x_val[0])
print(p_val.shape)
for i in p_val[0]:
    plt.scatter(i[0] * 256, i[1] * 256)
plt.savefig("test_data_gen_mpii.jpg")
