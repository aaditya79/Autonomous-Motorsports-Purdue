import cv2
img = cv2.imread('car.jpg',0)
img_normalized = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
cv2.imshow('Normalized Image - Model 1', img_normalized)
cv2.waitKey(0)
