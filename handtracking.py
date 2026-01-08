import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []

        base_options = python.BaseOptions(
            model_asset_path="hand_landmarker.task"
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.maxHands,
            min_hand_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
            min_hand_presence_confidence=self.trackCon
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imgRGB
        )

        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        self.results = self.detector.detect_for_video(mp_image, timestamp)

        if self.results.hand_landmarks and draw:
            for hand in self.results.hand_landmarks:
                for lm in hand:
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        bbox = []

        if self.results.hand_landmarks:
            hand = self.results.hand_landmarks[handNo]
            h, w, _ = img.shape

            xList, yList = [], []

            for id, lm in enumerate(hand):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(
                    img,
                    (xmin - 20, ymin - 20),
                    (xmax + 20, ymax + 20),
                    (0, 255, 0),
                    2
                )

        return self.lmList, bbox

    def fingersUp(self):
        if not self.lmList:
            return []

        fingers = []

        # Thumb
        fingers.append(
            1 if self.lmList[self.tipIds[0]][1] >
                 self.lmList[self.tipIds[0] - 1][1] else 0
        )

        # Other fingers
        for id in range(1, 5):
            fingers.append(
                1 if self.lmList[self.tipIds[id]][2] <
                     self.lmList[self.tipIds[id] - 2][2] else 0
            )

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
