import sys
import cv2
import time
import numpy as np
import threading
from collections import deque
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QCheckBox, QTextEdit, QSplitter, QDialog, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

from camera_stream import CameraStream
from detector import HumanDetector, PhoneDetector, FaceMeshDetector
from utils import estimate_distance, draw_overlay
from data_collector import DataCollector
from classifier import PhoneClassifier
from smoother import BoxSmoother
from db_manager import DBManager
from worker import DetectionWorker

class LogViewerDialog(QDialog):
    def __init__(self, logs):
        super().__init__()
        self.setWindowTitle("Database Logs (Last 50)")
        self.setGeometry(200, 200, 600, 400)
        
        layout = QVBoxLayout(self)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Event", "Timestamp", "Details", "Avg Score"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.table.setRowCount(len(logs))
        for row, log in enumerate(logs):
            
            self.table.setItem(row, 0, QTableWidgetItem(str(log[0])))
            self.table.setItem(row, 1, QTableWidgetItem(str(log[1])))
            self.table.setItem(row, 2, QTableWidgetItem(str(log[2])))
            self.table.setItem(row, 3, QTableWidgetItem(f"{log[3]:.2f}"))
            
        layout.addWidget(self.table)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

class HumanDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Human Detection System")
        self.setGeometry(100, 100, 950, 800) 

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget) 

        self.top_widget = QWidget()
        self.top_layout = QVBoxLayout(self.top_widget)
        
        self.bottom_widget = QWidget()
        self.bottom_layout = QVBoxLayout(self.bottom_widget)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.top_widget)
        self.splitter.addWidget(self.bottom_widget)
        self.splitter.setStretchFactor(0, 3) 
        self.splitter.setStretchFactor(1, 1)
        
        self.main_layout.addWidget(self.splitter)

        self.control_layout = QHBoxLayout()
        self.top_layout.addLayout(self.control_layout)

        self.camera_label = QLabel("Camera:")
        self.control_layout.addWidget(self.camera_label)

        self.camera_combo = QComboBox()
        self.available_cameras = self.get_available_cameras()
        self.camera_combo.addItems(self.available_cameras)
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        self.control_layout.addWidget(self.camera_combo)
        
        self.control_layout.addSpacing(10)
        
        self.auto_improve_chk = QCheckBox("Auto-Improve")
        self.auto_improve_chk.stateChanged.connect(lambda: self.log("Auto-Improve Toggled"))
        self.control_layout.addWidget(self.auto_improve_chk)
        
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.train_model)
        self.control_layout.addWidget(self.train_btn)
        
        self.clear_btn = QPushButton("Clear Data")
        self.clear_btn.setStyleSheet("color: white; background-color: #cc0000; font-weight: bold;")
        self.clear_btn.clicked.connect(self.clear_data)
        self.control_layout.addWidget(self.clear_btn)
        
        self.clear_db_btn = QPushButton("Clear DB")
        self.clear_db_btn.setStyleSheet("color: white; background-color: #cc0000; font-weight: bold;")
        self.clear_db_btn.clicked.connect(self.clear_database)
        self.control_layout.addWidget(self.clear_db_btn)
        
        self.control_layout.addStretch()
        
        self.readme_btn = QPushButton("Open Readme")
        self.readme_btn.clicked.connect(self.open_readme)
        self.control_layout.addWidget(self.readme_btn)
        
        self.history_btn = QPushButton("View Database")
        self.history_btn.clicked.connect(self.view_history)
        self.control_layout.addWidget(self.history_btn)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)
        self.control_layout.addWidget(self.exit_btn)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.top_layout.addWidget(self.video_label, 1)

        self.log_label = QLabel("System Logs")
        self.log_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 5px;")
        self.bottom_layout.addWidget(self.log_label)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e; 
                color: #dcdcdc; 
                font-family: 'Consolas', 'Monaco', monospace; 
                font-size: 12px;
                padding: 10px;
                border: 1px solid #333;
            }
        """)
        self.bottom_layout.addWidget(self.log_area)
        
        self.dev_label = QLabel()
        self.dev_label.setText('Developed by <a href="https://jenilsoni.vercel.app" style="color: #00ABF0; text-decoration: underline;">Jenil</a>')
        self.dev_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dev_label.setOpenExternalLinks(True)
        self.bottom_layout.addWidget(self.dev_label)

        self.log("Initializing system...")
        
        self.db = DBManager()
        if self.db.connected:
            self.log("Connected to SQLite Database.", color="#00ff00")
            self.db.log_system_start()
        else:
            self.log("Failed to connect to Database.", color="#ff0000")

        self.human_detector = None
        self.phone_detector = None
        self.face_mesh_detector = None
        self.data_collector = DataCollector()
        self.classifier = PhoneClassifier()
        
        self.face_smoother = BoxSmoother(alpha=0.4) 
        self.phone_smoother = BoxSmoother(alpha=0.4)
        
        try:
            self.human_detector = HumanDetector()
            self.log("HumanDetector loaded.", color="#00ff00")
        except Exception as e:
            self.log(f"Error loading HumanDetector: {e}", color="#ff0000")
            
        try:
            self.phone_detector = PhoneDetector()
            self.log("PhoneDetector loaded.", color="#00ff00")
        except Exception as e:
            self.log(f"Error loading PhoneDetector: {e}", color="#ff0000")
            
        try:
            self.face_mesh_detector = FaceMeshDetector()
            self.log("FaceMeshDetector loaded (Eyes).", color="#00ff00")
        except Exception as e:
            self.log(f"Error loading FaceMeshDetector: {e}", color="#ff0000")
        
        if self.classifier.trained:
             self.log("SVM Model loaded successfully.", color="#00aa00")
        
        self.worker = DetectionWorker(self.human_detector, self.phone_detector, self.face_mesh_detector)
        self.worker.frame_processed.connect(self.on_detection_complete)
        self.worker.start()
        
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.prev_time = 0
        self.frame_count = 0
        
        self.last_results = None
        self.last_phone_results = None
        self.last_face_results = None 
        self.last_scale_factor = 1.0 
        self.last_confidence = 0.0
        self.last_distance = "Unknown"
        self.svm_verified = False
        self.current_fps = 0
        self.eyes_closed_state = False 
        
        self.last_face_status = "Unknown" 
        self.no_face_log_time = 0
        self.last_mobile_log_time = 0
        
        self.face_history = deque()
        self.phone_history = deque()
        self.eyes_history = deque() 
        self.distance_history = deque() 
        
        self.db_logged_no_face = False
        self.db_logged_phone = False
        self.db_logged_eyes = False
        self.db_logged_far = False

        self.start_camera(0)
        
    def log(self, message, color=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if color:
            formatted_msg = f'<span style="color:{color}">[{timestamp}] {message}</span>'
        else:
            formatted_msg = f'[{timestamp}] {message}'
            
        self.log_area.append(formatted_msg)
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def get_available_cameras(self):
        checks = 3
        available = []
        for i in range(checks):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(f"Camera {i}")
                cap.release()
        return available if available else ["Camera 0"]

    def start_camera(self, index):
        if self.camera:
            self.camera.stop()
            self.timer.stop()
        
        self.log(f"Starting camera {index}...")
        self.camera = CameraStream(src=index).start()
        self.timer.start(30) 

    def change_camera(self, index):
        selection = self.camera_combo.currentText()
        try:
            cam_idx = int(selection.split(" ")[1])
            self.start_camera(cam_idx)
        except:
            pass
            
    def format_size(self, size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.2f} KB"
        else:
            return f"{size_bytes/(1024*1024):.2f} MB"

    def clear_data(self):
        self.log("Clearing environment data...", color="#ffaa00")
        self.data_collector.clear_data()
        self.classifier.clear_model()
        self.log("All training data and models cleared.", color="#ff0000")

    def clear_database(self):
        self.log("Clearing database logs...", color="#ffaa00")
        if self.db.clear_logs():
            self.log("Database logs cleared successfully.", color="#ff0000")
        else:
            self.log("Failed to clear database.", color="#ff0000")

    def train_model(self):
        self.train_btn.setEnabled(False)
        self.train_btn.setText("Training...")
        self.log("Starting model training...", color="#00ffff")
        
        def run_training():
            try:
                success = self.classifier.train()
                if success:
                     model_size = self.classifier.get_model_size()
                     data_size = self.data_collector.get_dataset_size()
                     
                     m_str = self.format_size(model_size)
                     d_str = self.format_size(data_size)
                     
                     self.log(f"Training Successful! (Model: {m_str}, Data: {d_str})", color="#00ff00")
                else:
                     self.log("Training Failed. No data?", color="#ff0000")
            except Exception as e:
                self.log(f"Training Error: {e}", color="#ff0000")
                
            self.train_btn.setText("Train Model")
            self.train_btn.setEnabled(True)
            
        threading.Thread(target=run_training).start()
        
    def open_readme(self):
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        import os
        from utils import get_resource_path
        
        # Try to find the readme. In exe, it might be in temp folder.
        readme_path = get_resource_path("README.md")
        
        # If not found (e.g. forgot to bundle), try local dir
        if not os.path.exists(readme_path):
             readme_path = os.path.abspath("README.md")
             
        QDesktopServices.openUrl(QUrl.fromLocalFile(readme_path))

    def view_history(self):
        if not self.db or not self.db.connected:
            self.log("Database not connected.", color="#ff0000")
            return
            
        logs = self.db.fetch_logs(50)
        if not logs:
             self.log("No logs found in DB.", color="#aaaa00")
        
        dialog = LogViewerDialog(logs)
        dialog.exec()
        
    def _update_history(self, history_deque, confidence, current_time, window_size=15.0):
        history_deque.append((current_time, confidence))
        while history_deque and (current_time - history_deque[0][0] > window_size):
            history_deque.popleft()
            
    def _get_avg_confidence(self, history_deque, duration=None):
        if not history_deque: return 0.0
        if duration is None:
            confs = [item[1] for item in history_deque]
            return sum(confs) / len(confs)
        else:
            latest_time = history_deque[-1][0]
            cutoff = latest_time - duration
            confs = [item[1] for item in history_deque if item[0] >= cutoff]
            if not confs: return 0.0
            return sum(confs) / len(confs)
        
    def _get_history_duration(self, history_deque):
        if not history_deque: return 0.0
        return history_deque[-1][0] - history_deque[0][0]

    def on_detection_complete(self, frame, results, confidence, phone_results, face_results, scale):
        self.last_results = results
        self.last_confidence = confidence
        self.last_phone_results = phone_results
        self.last_face_results = face_results 
        self.last_scale_factor = scale
        
        current_time = time.time()
        
        current_face_conf = confidence if confidence else 0.0
        human_detected = False
        
        if results and results.pose_landmarks:
            human_detected = True
            height, width, _ = frame.shape
            landmarks = results.pose_landmarks[0]
            face_landmarks = landmarks[:11]
            
            y_min, y_max = height, 0
            for lm in face_landmarks:
                y = int(lm.y * height)
                if y < y_min: y_min = y
                if y > y_max: y_max = y
            
            bbox_h = y_max - y_min
            self.last_distance = estimate_distance(bbox_h, height)
            
            if self.last_face_status != "Detected":
                self.log("Face Detected", color="#00ff00")
                self.last_face_status = "Detected"
        else:
            self.last_distance = "Unknown"
            if self.last_face_status != "None":
                self.log("Face Lost", color="#ffaa00")
                self.last_face_status = "None"
            
            if current_time - self.no_face_log_time > 5.0:
                self.log("No Face Detected", color="#888888")
                self.no_face_log_time = current_time
            
            self.face_smoother.reset()

        self._update_history(self.face_history, current_face_conf, current_time)
        face_duration = self._get_history_duration(self.face_history)
        face_avg = self._get_avg_confidence(self.face_history)
        
        if face_duration >= 14.0 and face_avg < 0.25:
            if not self.db_logged_no_face:
                self.db.log_no_face(avg_accuracy=face_avg)
                self.log("DB: Logged No Face (15s Avg)", color="#ff00ff")
                self.db_logged_no_face = True
        else:
            if self.db_logged_no_face:
                recent_avg = self._get_avg_confidence(self.face_history, duration=5.0)
                if recent_avg > 0.5:
                     self.db.log_face_recovered(avg_accuracy=recent_avg)
                     self.log("DB: Logged Face Recovered (5s Avg)", color="#00ff00")
                     self.db_logged_no_face = False
            elif face_avg > 0.6: 
                 self.db_logged_no_face = False
                 
        is_far = 1.0 if self.last_distance in ["Far", "Very Far"] else 0.0
        if not human_detected: is_far = 0.0 
        
        self._update_history(self.distance_history, is_far, current_time)
        dist_duration = self._get_history_duration(self.distance_history)
        far_avg = self._get_avg_confidence(self.distance_history)
        
        if dist_duration >= 14.0 and far_avg > 0.8: 
             if not self.db_logged_far:
                 self.db.log_too_far(avg_accuracy=far_avg)
                 self.log("DB: Logged Person Too Far (15s Avg)", color="#ff00ff")
                 self.db_logged_far = True
        else:
            if self.db_logged_far:
                 recent_far_avg = self._get_avg_confidence(self.distance_history, duration=5.0)
                 if recent_far_avg < 0.2:
                     self.db.log_back_in_range(avg_accuracy=recent_far_avg)
                     self.log("DB: Logged Back in Range (5s Avg)", color="#00ff00")
                     self.db_logged_far = False
            elif far_avg < 0.5:
                 self.db_logged_far = False

        current_phone_conf = 0.0
        if not phone_results or not phone_results.detections:
            self.phone_smoother.reset()
        else:
            scores = [d.categories[0].score for d in phone_results.detections]
            if scores:
                current_phone_conf = max(scores)

        self.svm_verified = False
        if phone_results and phone_results.detections:
            for detection in phone_results.detections:
                bbox_norm = detection.bounding_box
                x = int(bbox_norm.origin_x / scale)
                y = int(bbox_norm.origin_y / scale)
                w = int(bbox_norm.width / scale)
                h = int(bbox_norm.height / scale)
                bbox = (x, y, w, h)
                score = detection.categories[0].score
                
                if current_time - self.last_mobile_log_time > 5.0:
                    self.log(f"Mobile Detected (Score: {int(score*100)}%)", color="#00ffff")
                    self.last_mobile_log_time = current_time
                
                if self.auto_improve_chk.isChecked() and score > 0.60:
                    self.data_collector.save_positive(frame, bbox)
                    self.data_collector.save_negative(frame, bbox)
                    
                if self.classifier.trained:
                    is_phone = self.classifier.predict(frame, bbox)
                    if is_phone == 1:
                        self.svm_verified = True
                        if self.frame_count % 60 == 0:
                            self.log("SVM Verified Phone Detection", color="#FFD700")

        self._update_history(self.phone_history, current_phone_conf, current_time)
        phone_duration = self._get_history_duration(self.phone_history)
        phone_avg = self._get_avg_confidence(self.phone_history)
        
        if phone_duration >= 14.0 and phone_avg > 0.40:
            if not self.db_logged_phone:
                self.db.log_phone_detected(avg_accuracy=phone_avg)
                self.log("DB: Logged Phone Detected (15s Avg)", color="#ff00ff")
                self.db_logged_phone = True
        else:
            if self.db_logged_phone:
                recent_avg = self._get_avg_confidence(self.phone_history, duration=5.0)
                if recent_avg < 0.15:
                    self.db.log_phone_removed(avg_accuracy=recent_avg)
                    self.log("DB: Logged Phone Removed (5s Avg)", color="#ffaa00")
                    self.db_logged_phone = False
            elif phone_avg < 0.25:
                 self.db_logged_phone = False
                 
        current_eyes_closed = 0.0
        if face_results and face_results.face_blendshapes:
            shapes = face_results.face_blendshapes[0] 
            left_blink = 0.0
            right_blink = 0.0
            
            for cat in shapes:
                if cat.category_name == 'eyeBlinkLeft':
                    left_blink = cat.score
                elif cat.category_name == 'eyeBlinkRight':
                    right_blink = cat.score
            
            current_eyes_closed = (left_blink + right_blink) / 2.0
        else:
            current_eyes_closed = 0.0
            
        self.eyes_closed_state = current_eyes_closed > 0.6
        
        self._update_history(self.eyes_history, current_eyes_closed, current_time)
        eyes_duration = self._get_history_duration(self.eyes_history)
        eyes_avg = self._get_avg_confidence(self.eyes_history)
        
        if eyes_duration >= 14.0 and eyes_avg > 0.6: 
            if not self.db_logged_eyes:
                 self.db.log_eyes_closed(avg_accuracy=eyes_avg)
                 self.log("DB: Logged Eyes Closed (15s Avg)", color="#ff00ff")
                 self.db_logged_eyes = True
        else:
             if self.db_logged_eyes:
                 recent_eye_avg = self._get_avg_confidence(self.eyes_history, duration=5.0)
                 if recent_eye_avg < 0.3: 
                     self.db.log_eyes_opened(avg_accuracy=recent_eye_avg)
                     self.log("DB: Logged Eyes Opened (5s Avg)", color="#00ff00")
                     self.db_logged_eyes = False
             elif eyes_avg < 0.4:
                 self.db_logged_eyes = False

    def update_frame(self):
        frame = self.camera.read()
        if frame is not None:
            current_time = time.time()
            self.current_fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = current_time
            
            self.worker.process_frame(frame)
            
            frame_drawn = draw_overlay(frame.copy(), self.last_results, self.current_fps, self.last_confidence, 
                                 self.last_distance, self.last_phone_results, self.svm_verified,
                                 face_smoother=self.face_smoother, phone_smoother=self.phone_smoother,
                                 scale_factor=self.last_scale_factor, eyes_closed=self.eyes_closed_state)
            
            frame_rgb = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap)
            
            self.frame_count += 1
    
    def closeEvent(self, event):
        self.log("Closing application...")
        if hasattr(self, 'worker'):
            self.worker.stop()
        if self.camera:
            self.camera.stop()
        if hasattr(self, 'db'):
            self.db.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HumanDetectionApp()
    window.show()
    sys.exit(app.exec())
