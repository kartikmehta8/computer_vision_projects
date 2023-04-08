import cv2 as cv
import mediapipe as mp
import time
import math

class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # DetectionCon & TrackCon not used
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth)


    def findPose(self, img, draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
        
    
    def findPosition(self, img, draw=True):

        self.landmarkList = []

        if self.results.pose_landmarks:
            for id, lndmrk in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lndmrk)

                cx, cy = int(lndmrk.x*w), int(lndmrk.y*h)
                self.landmarkList.append([id,cx,cy])
                if draw:
                    cv.circle(img, (cx,cy), 5, (255,0,0), cv.FILLED)

        return self.landmarkList
    
    def findAngle(self, img, p1, p2, p3, draw=True):
        
        # Landmarks
        x1, y1 = self.landmarkList[p1][1:]
        x2, y2 = self.landmarkList[p2][1:]
        x3, y3 = self.landmarkList[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2)-math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle = angle+360

        # print(angle)

        if draw:
            cv.line(img, (x1,y1), (x2,y2), (0,255,0), 3)
            cv.line(img, (x2,y2), (x3,y3), (0,255,0), 3)
            cv.circle(img, (x1,y1), 10, (0,0,255), cv.FILLED)
            cv.circle(img, (x1,y1), 15, (0,0,255), 2)
            cv.circle(img, (x2,y2), 10, (0,0,255), cv.FILLED)
            cv.circle(img, (x2,y2), 15, (0,0,255), 2)
            cv.circle(img, (x3,y3), 10, (0,0,255), cv.FILLED)
            cv.circle(img, (x3,y3), 15, (0,0,255), 2)
            cv.putText(img, str(int(angle)), (x2+40,y2+50), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

        return angle

def main():

    cap = cv.VideoCapture('videos/video1.mp4')
    pTime = 0

    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        landmarkList = detector.findPosition(img)
        
        if len(landmarkList) != 0:
            print(landmarkList)

            # 14 for ex - is a type of body part number (for tracking)
            cv.circle(img, (landmarkList[4][1], landmarkList[4][2]), 15, (0,0,255), cv.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv.imshow('Video', img)

        cv.waitKey(5)

if __name__ == '__main__':
    main()