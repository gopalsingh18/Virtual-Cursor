import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceDetector:
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path="face_landmarker.task"
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

    def get_nose_y(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        results = self.detector.detect_for_video(mp_image, timestamp)

        if results.face_landmarks:
            nose = results.face_landmarks[0][1]  # SAME index as old FaceMesh
            h, _, _ = img.shape
            return int(nose.y * h)

        return None
