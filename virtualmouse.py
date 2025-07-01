from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key
import keyboard as kb  # For volume control
import cv2
import numpy as np
import handtracking as htm
import time
import mediapipe as mp
import os

##########################
# Camera and tracking settings
wCam, hCam = 640, 480
frameR = 100
smoothening = 7
doubleClickThreshold = 0.5
eyeLockThreshold = 4
pauseThreshold = 1
volumeSensitivity = 4
##########################

# Variables
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
lastClickTime = 0
lastEyeDetectedTime = time.time()
mediaPaused = False

# Initialize mouse and keyboard controllers
mouse = MouseController()
keyboard = KeyboardController()

# Initialize camera
cap = cv2.VideoCapture(0)  # Use your default camera
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector
detector = htm.handDetector(maxHands=1, detectionCon=0.7, trackCon=0.7)

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Track nose Y-position for volume control
prev_nose_y = None

def toggle_media():
    """Simulates Play/Pause by pressing spacebar."""
    keyboard.press(Key.space)
    keyboard.release(Key.space)

def change_volume(direction):
    """Changes system volume by simulating volume keys."""
    if direction == "up":
        kb.send("volume up")
    else:
        kb.send("volume down")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame.")
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    results = face_mesh.process(imgRGB)
    eye_detected = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            eye_detected = True
            lastEyeDetectedTime = time.time()

            nose = face_landmarks.landmark[1]
            nose_y = int(nose.y * hCam)

            if prev_nose_y is not None:
                movement = nose_y - prev_nose_y
                if movement > volumeSensitivity:
                    print("Head down: Volume down")
                    change_volume("down")
                elif movement < -volumeSensitivity:
                    print("Head up: Volume up")
                    change_volume("up")

            prev_nose_y = nose_y

    if not eye_detected and (time.time() - lastEyeDetectedTime > eyeLockThreshold):
        print("No eyes detected. Locking screen...")
        os.system("rundll32.exe user32.dll,LockWorkStation")  # Windows lock

    if not eye_detected and (time.time() - lastEyeDetectedTime > pauseThreshold):
        if not mediaPaused:
            print("No eyes detected. Pausing media...")
            toggle_media()
            mediaPaused = True

    if eye_detected and mediaPaused:
        print("Eyes detected again. Resuming media...")
        toggle_media()
        mediaPaused = False

    # Hand detection
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        fingers = detector.fingersUp()

        if len(fingers) >= 5:
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, 1920))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, 1080))

                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                mouse.position = (clocX, clocY)

                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 12, img)
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    currentTime = time.time()
                    if currentTime - lastClickTime < doubleClickThreshold:
                        mouse.click(Button.left, 2)
                        lastClickTime = 0
                    else:
                        mouse.click(Button.left, 1)
                        lastClickTime = currentTime

            if fingers[0] == 1 and sum(fingers[1:]) == 0:
                mouse.click(Button.right, 1)
                time.sleep(1)

            if fingers[3] == 1 and fingers[4] == 0:
                mouse.scroll(0, -1)

            if fingers[4] == 1 and fingers[3] == 0:
                mouse.scroll(0, 1)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
