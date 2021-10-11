import sys, os
import cv2 as cv

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=False)

img = x_train[0][0]
print(img.shape)

cv.imshow('image', img)
cv.waitKey()

cv.destroyAllWindows()
