import cv2 as cv
import time
import os
from handTrackingModule import handDetector

######################
wCam, hCam = 640, 480
######################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


prevTime = 0

detector = handDetector()

tipIds = [4,8,12,16,20]

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lndmrkList = detector.findPosition(img, draw=False)
    # print(lndmrkList)

    if len(lndmrkList) != 0:
        fingers = []

        # Thumb
        if lndmrkList[tipIds[0]][1] > lndmrkList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):

            if lndmrkList[tipIds[id]][2] < lndmrkList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        totalFingers = fingers.count(1)
        cv.putText(img, str(totalFingers), (20,130), cv.FONT_HERSHEY_PLAIN, 4, (0,255,0), 3)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 2)

    cv.imshow('Video', img)
    cv.waitKey(1)