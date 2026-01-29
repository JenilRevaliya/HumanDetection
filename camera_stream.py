import cv2
import threading
import time

class CameraStream:
    
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.stream.isOpened():
            print("Error: Could not open camera.")
            self.stopped = True
            return

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        
        if self.stopped:
            return self
            
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        
        while not self.stopped:
            if not self.stream.isOpened():
                self.stop()
                break
                
            (grabbed, frame) = self.stream.read()
            if grabbed:
                self.frame = frame
            else:
                self.stop()

    def read(self):
        
        return self.frame

    def stop(self):
        
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()
