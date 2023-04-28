import cv2
import numpy as np
makeOrb = cv2.ORB_create(nfeatures = 1000)
firstImage = cv2.imread('/Users/aadityapai/Desktop/Work/Fall2022/VIP47921/image1.jpg',1)
secondImage = cv2.imread('/Users/aadityapai/Desktop/Work/Fall2022/VIP47921/image1.jpg',1)
firstImageConv = cv2.cvtColor(firstImage, cv2.COLOR_RGB2GRAY)
secondImageConv = cv2.cvtColor(secondImage, cv2.COLOR_RGB2GRAY)
trainPoints, trainDetect = makeOrb.detectAndCompute(firstImageConv, None)
testPoints, testDetect = makeOrb.detectAndCompute(secondImageConv, None)
copyFirstImage = np.copy(firstImage)
cv2.drawKeypoints(firstImage, trainPoints, copyFirstImage, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
fPoint = np.copy(firstImage)
cv2.drawKeypoints(firstImage, trainPoints, fPoint, flags=2)
norma = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
success = norma.match(trainDetect, testDetect)
success = sorted(success, key = lambda x : x.distance)
exact = success[:100]
h, w = firstImageConv.shape[:2]
imagePoints = np.float32([[0, 0],[0, h - 1],[w - 1, h - 1],[w - 1, 0] ]).reshape(-1, 1, 2)


training = np.float32([trainPoints[m.queryIdx].pt for m in exact]).reshape(-1,1,2)
testing = np.float32([testPoints[m.trainIdx].pt for m in exact]).reshape(-1,1,2)
z, mm = cv2.findHomography(training, testing, cv2.RANSAC,5.0)
cv2.imshow('Areas of Interest', copyFirstImage)
fPoint = cv2.drawMatches(fPoint,trainPoints,secondImage,testPoints,exact, None,flags=2)
cv2.imshow('Matching Through Comparison', fPoint)
transformPoints = cv2.perspectiveTransform(imagePoints,z)
res = cv2.polylines(secondImage, [np.int32(transformPoints)], True, (50,0,255), 3, cv2.LINE_AA)
cv2.imshow('Object in View', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
