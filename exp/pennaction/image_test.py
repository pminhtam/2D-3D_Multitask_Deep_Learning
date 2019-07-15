import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

from PIL import Image

from deephar.utils import *
import matplotlib.pyplot as plt

image = T(Image.open("aa.jpg"))

image.rotate_crop(0, (250,250), (500,500))
image.resize((250,250))
plt.imshow(image.asarray())
plt.savefig("img.jpg")

print(image.asarray())