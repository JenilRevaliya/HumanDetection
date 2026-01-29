import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QObject

class DetectionWorker(QThread):
    frame_processed = pyqtSignal(np.ndarray, object, float, object, object, float)

    def __init__(self, human_detector, phone_detector, face_mesh_detector):
        super().__init__()
        self.human_detector = human_detector
        self.phone_detector = phone_detector
        self.face_mesh_detector = face_mesh_detector
        self.running = True
        self.current_frame = None
        self.is_busy = False

    def process_frame(self, frame):
        
        if not self.is_busy:
            self.current_frame = frame.copy()

    def run(self):
        while self.running:
            if self.current_frame is not None:
                self.is_busy = True
                frame = self.current_frame
                self.current_frame = None # Clear it so we don't re-process
                
                h, w = frame.shape[:2]
                target_w = 640
                scale = 1.0
                
                small_frame = frame
                if w > target_w:
                    scale = target_w / w
                    target_h = int(h * scale)
                    small_frame = cv2.resize(frame, (target_w, target_h))
                else:
                    scale = 1.0 # No downscaling needed if already small
                
                pose_results = None
                pose_conf = 0.0
                phone_results = None
                face_results = None
                
                if self.human_detector:
                     pose_results, pose_conf = self.human_detector.detect(small_frame)
                
                if self.phone_detector:
                     phone_results = self.phone_detector.detect(small_frame)
                     
                if self.face_mesh_detector and pose_results and pose_results.pose_landmarks:
                     face_results = self.face_mesh_detector.detect(small_frame)
                
                self.frame_processed.emit(frame, pose_results, pose_conf, phone_results, face_results, scale)
                self.is_busy = False
            else:
                self.msleep(10) # Sleep briefly to avoid CPU spin

    def stop(self):
        self.running = False
        self.wait()
