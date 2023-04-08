import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands

        # Not providing detectionCon & trackCon
        self.hands = self.mpHands.Hands(self.mode, self.maxHands)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
                

    def findPosition(self, img, handNumber=0, draw=True):

        landmarkList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]

            for id, lndmrk in enumerate(myHand.landmark):
                h, w, channels = img.shape

                # Center
                cx, cy = int(lndmrk.x*w), int(lndmrk.y*h)
                # print(id, cx, cy)
                landmarkList.append([id,cx,cy])

                if draw:
                    cv.circle(img, (cx, cy), 7, (255,0,0), cv.FILLED)
        
        return landmarkList


def main():

    prevTime = 0
    currTime = 0
    cap = cv.VideoCapture(0)

    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img)
        
        if len(landmarkList) != 0:
            print(landmarkList[4])

        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime

        cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 2)

        cv.imshow('Video', img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()