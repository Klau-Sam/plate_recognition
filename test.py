# Importing all required packages
import cv2
import numpy as np
import matplotlib.pyplot as plt  # matplotlib inline

# Read in the cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# russian_plate = cv2.CascadeClassifier('haarcascade_licence_rus.xml')
russian_plate = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")


def adjusted_russian_plate(img):
    russian_ing = img.copy()

    face_rect = face_cascade.detectMultiScale(russian_ing, scaleFactor=1.05, minNeighbors=20)

    for (x, y, w, h) in face_rect:
        cv2.rectangle(russian_ing, (x, y),
                      (x + w, y + h), (255, 255, 255), 10)

    return russian_ing


# create a function to detect face
def adjusted_detect_face(img):
    face_img = img.copy()

    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in face_rect:
        cv2.rectangle(face_img, (x, y),
                      (x + w, y + h), (255, 255, 255), 10)

    return face_img


# create a function to detect eyes
def detect_eyes(img):
    eye_img = img.copy()
    eye_rect = eye_cascade.detectMultiScale(eye_img, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in eye_rect:
        cv2.rectangle(eye_img, (x, y),
                      (x + w, y + h), (255, 255, 255), 10)
    return eye_img


# Reading in the image and creating copies
img = cv2.imread('car0.jpg')
if img is None:
    print("read file error")

img_copy1 = img.copy()
img_copy2 = img.copy()
img_copy3 = img.copy()

# eyes_face = adjusted_detect_face(img_copy3)
# eyes_face = detect_eyes(eyes_face)
# plt.imshow(eyes_face)

car = adjusted_russian_plate(img_copy3)

plt.imshow(car)
plt.show()
print("2")
