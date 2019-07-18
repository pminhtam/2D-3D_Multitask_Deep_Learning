import matplotlib.pyplot as plt
import cv2

# img = cv2.imread("datasets/MPII/images/086617615.jpg")
# img = cv2.imread("/mnt/hdd10tb/Users/tam/COCO/val2017/000000015335.jpg")
img = cv2.imread("/mnt/hdd10tb/Users/duong/MERL/31_2_crop_2688_2730_RetractFromShelf-0001.jpg")
img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

plt.imshow(img)
plt.savefig("test.jpg")
