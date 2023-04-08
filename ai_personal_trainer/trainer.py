import cv2 as cv
import numpy as np
import time
from poseModule import poseDetector

######################
wCam, hCam = 640, 480
######################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

prevTime = 0

detector = poseDetector()
count = 0
dir = 0

while True:

    success, img = cap.read()

    img = detector.findPose(img, draw=False)

    lndmrkList = detector.findPosition(img, draw=False)
    # print(lndmrkList)

    if len(lndmrkList) != 0:
        # Right hand
        # detector.findAngle(img, 12,14,16)
        # Left hand
        angle = detector.findAngle(img, 11,13,15)
        per = np.interp(angle, (210, 310), (0,100))
        bar = np.interp(angle, (220, 310), (650,100))


        # Check for the dumbbell curl
        if per == 100:
            if dir == 0:
                count = count + 0.5
                dir = 1
        if per == 0:
            if dir == 1:
                count = count + 0.5
                dir = 0
        # print(count)

        # Draw curle count
        cv.putText(img, f'{int(count)}', (50,120), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)


    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime
    cv.putText(img, str(int(fps)), (10,50), cv.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

    cv.imshow('Video', img)
    cv.waitKey(1)