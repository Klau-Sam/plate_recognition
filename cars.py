import cv2
import numpy as np
from skimage import data, io, filters,feature,morphology
from skimage.morphology import disk, ball, square
img = cv2.imread("car0.jpg", 1)
#img = cv2.GaussianBlur(img, (5, 5), 0)  # blur
#img = cv2.Canny(img, 100, 200)  # get Canny edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = filters.median(gray, disk(1))
#gray = filters.sobel(gray)
#gray = morphology.erosion(gray, disk(5))
#gray = morphology.dilation(gray, disk(2))
cv2.imshow('gray', gray)
cv2.waitKey(0)
#img = cv2.GaussianBlur(img, (5, 5), 0)  # blur
plates_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
plates = plates_cascade.detectMultiScale(gray,  scaleFactor=1.05, minNeighbors=10)
for (x,y,w,h) in plates:
    plates_rec = cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 5)
    #cv2.putText(plates_rec, 'Text', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imshow('img', img)
cv2.waitKey(0)
print('Number of detected licence plates:', len(plates))