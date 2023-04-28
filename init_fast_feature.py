import numpy as np
import cv2

image = cv2.imread('/Users/aadityapai/Desktop/Work/Fall2022/VIP47921/image1.jpg')
detectF = cv2.FastFeatureDetector_create()
detectF.setNonmaxSuppression(False)
imageConv = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
k = detectF.detect(imageConv, None)
copy = np.copy(image)
cv2.drawKeypoints(image, k, copy, color=(100, 50, 200))
cv2.imshow('Feature Mapping Algorithm', copy)
cv2.waitKey(0)
