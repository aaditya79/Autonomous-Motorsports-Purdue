import cv2

video = cv2.VideoCapture(0)
video.set(4, 720)
video.set(5, 520)
faceCap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
   prob, image = video.read()
   imageConv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   faceCap2 = faceCap.detectMultiScale(imageConv, 1.3, 6)
   for (i, j, k, l) in faceCap2:
       image = cv2.rectangle(image, (i, j), (i + k, j + l), (100, 50, 200), 5)
   cv2.imshow('Face Detection using Haar Cascades', image)
   if cv2.waitKey(10) & 0xFF == ord('l'):
       break
  
video.release()
cv2.destroyWindow('Face Detection using Haar Cascades')
