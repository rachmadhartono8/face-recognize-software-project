import math

import cv2
# import torch
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import time

# print(torch.cuda.is_available())

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5","model/labels.txt")
offset = 20
imgSize = 300

labels = ["A","B","C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand= hands[0]
        x, y, w, h = hand['bbox']

        y1 = y - offset
        y2 = y + h + offset
        x1 = x - offset
        x2 = x + w + offset
        x1, x2, y1, y2 = x1 * (x1>0), x2 * (x2>0), y1 * (y1>0), y2 * (y2>0)
        # print(x1, x2, y1, y2)
        imgCrop = img[y1:y2, x1:x2]
        # cv2.imshow("image crop", imgCrop)

        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255
        imgCropShape = imgCrop.shape


        aspecRatio = h/w

        if aspecRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgCrop.shape
            wGap =math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wGap+wCal] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize,hCal))
            imgResizeShape = imgCrop.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        prediction, index = classifier.getPrediction(img)
        print(prediction,index)

        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        # cv2.imshow("image white", imgWhite)

    cv2.imshow("image",imgOutput)
    cv2.waitKey(1)


