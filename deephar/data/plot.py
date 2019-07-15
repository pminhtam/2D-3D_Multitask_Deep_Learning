import matplotlib.pyplot as plt
import cv2

img = cv2.imread("datasets/MPII/images/086617615.jpg")
img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

plt.imshow(img)
plt.savefig("test.jpg")
