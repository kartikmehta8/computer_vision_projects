import cv2 as cv
import numpy as np
import time
from handTrackingModule import handDetector
import autopy
import math

#################
wCam, hCam = 640, 480
frameR = 100
smoothening = 5
pLocX, pLocY = 0,0
cLocX, cLocY = 0,0
#################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = handDetector(maxHands=1)

wScr, hScr = autopy.screen.size()
print(wScr, hScr)

prevTime = 0

while True:

    # 1. Find hand landmarks
    success, img = cap.read()

    img = detector.findHands(img)

    lndmrkList = detector.findPosition(img)

    # 2. Get the tip of index and middle finger
    if len(lndmrkList) != 0:
        x1, y1 = lndmrkList[8][1:]
        x2, y2 = lndmrkList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        cv.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (0,0,255), 2)

        # 4. Only index finger : moving mode
        if fingers[1]==1 and fingers[2]==0:

            # 5. Convert coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))


            # 6. Smoothen values
            cLocX = pLocX+(x3-pLocX)/smoothening
            cLocY = pLocY+(x3-pLocY)/smoothening

            # 7. Move mouse
            autopy.mouse.move(wScr-cLocX,cLocY)
            cv.circle(img, (x1,y1), 15, (255,0,255), cv.FILLED)

            pLocX, pLocY = cLocX, cLocY

        # 8. Both finger up : clicking mode
        if fingers[1]==1 and fingers[2]==1:
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # 9. Find distance between fingers
            if length < 40:
                cv.circle(img, (x1,y1), 15, (0,255,0), cv.FILLED)
                autopy.mouse.click()

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv.putText(img, str(int(fps)), (20,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 2)

    cv.imshow('Video', img)
    cv.waitKey(1)