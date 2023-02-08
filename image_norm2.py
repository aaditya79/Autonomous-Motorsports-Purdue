import cv2
img = cv2.imread('car.jpg',0)
classifier = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
img_normalized = cv2.normalize(classifier, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
cv2.imshow('Normalized Image - Model 2', img_normalized)
cv2.waitKey(0)
