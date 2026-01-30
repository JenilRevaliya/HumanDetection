import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


from utils import get_resource_path

class HumanDetector:
    
    def __init__(self, model_path="pose_landmarker_lite.task"):
        model_path = get_resource_path(model_path)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False)
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame):
        
        if frame is None:
            return None, 0.0

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        detection_result = self.landmarker.detect(mp_image)
        
        confidence = 0.0
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            visibilities = [lm.visibility for lm in landmarks[:5]]
            confidence = sum(visibilities) / len(visibilities) if visibilities else 0.0

        return detection_result, confidence

class PhoneDetector:
    
    def __init__(self, model_path="efficientdet_lite0.tflite"):
        model_path = get_resource_path(model_path)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.5, # Adjust as needed
            category_allowlist=["cell phone"] # Only detect phones
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

    def detect(self, frame):
        
        if frame is None:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        return self.detector.detect(mp_image)

class FaceMeshDetector:
    
    def __init__(self, model_path="face_landmarker.task"):
        import os
        model_path = get_resource_path(model_path) # Check local/MEIPASS first
        
        if not os.path.exists(model_path):
            # Fallback: model might not be in MEIPASS but we want to download it locally?
            # In an exe, we can't write to MEIPASS. We should write to CWD or AppData.
            # For simplicity, if it's missing in Exe, we might fail or write to current dir.
            # But get_resource_path returns MEIPASS path if packaged.
            
            # If we are here, it means file is NOT in the bundle. 
            # We should try to download to the current directory (local execution)
            # OR checking if it exists in current directory explicitly.
            
            local_path = "face_landmarker.task"
            if not os.path.exists(local_path):
                 print(f"Downloading {local_path}...")
                 url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                 import urllib.request
                 try:
                     urllib.request.urlretrieve(url, local_path)
                     print("Download complete.")
                 except Exception as e:
                     print(f"Failed to download model: {e}")
            
            # Use the local path if the bundled one failed
            if os.path.exists(local_path):
                model_path = local_path

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame):
        
        if frame is None: return None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        return self.landmarker.detect(mp_image)
