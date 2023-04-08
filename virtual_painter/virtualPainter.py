import cv2 as cv
import time
import time
import os
from handTrackingModule import handDetector
import numpy as np

folderPath = 'header'
myList = os.listdir(folderPath)
overlayList = []
brushThickness = 20
eraserThickness = 60

for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (255,0,255)


cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = handDetector()

xp, yp = 0,0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    # 1. Import Image
    success, img = cap.read()
    img = cv.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)

    lndmrkList = detector.findPosition(img, draw=False)

    if len(lndmrkList) != 0:

        # Tip of index & middle finger
        x1, y1 = lndmrkList[8][1:]
        x2, y2 = lndmrkList[12][1:]

        # 3. Check which fingers are up

        fingers = detector.fingersUp()

        # 4. If selection mode - Two fingers up
        if fingers[1] and fingers[2]:
            
            xp, yp = 0,0

            cv.rectangle(img, (x1,y1-25), (x2,y2+25), (255,0,255), cv.FILLED)
            if y1 < 125:
                if 250<x1<450:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif 550<x1<750:
                    header = overlayList[3]
                    drawColor = (255,0,0)
                elif 800<x1<950:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif 1000<x1:
                    header = overlayList[1]
                    drawColor=(0,0,0)
            
         # 5. If drawing mode - Index finger up
        if fingers[1] and fingers[2]==False:
            cv.circle(img, (x1,y1), 15, drawColor, cv.FILLED)

            if xp == 0 and yp == 0:
                xp,yp = x1,y1
                 
            if drawColor == (0,0,0):
                cv.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

            cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    # MAGIC
    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgCanvas)


    # Header image
    img[0:125, 0:1280] = header

    # img = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv.imshow('Video Game', img)
    # cv.imshow('Video Canvas', imgCanvas)
    cv.waitKey(1)