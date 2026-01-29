import cv2
import os
import time
import random
import numpy as np

class DataCollector:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.pos_dir = os.path.join(data_dir, "positives")
        self.neg_dir = os.path.join(data_dir, "negatives")
        
        os.makedirs(self.pos_dir, exist_ok=True)
        os.makedirs(self.neg_dir, exist_ok=True)
        
        self.count_limit = 500 # Limit samples to avoid bloat
        
    def save_positive(self, frame, bbox):
        
        if frame is None: return
        
        x, y, w, h = bbox
        h_img, w_img, _ = frame.shape
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w < 10 or h < 10: return
        
        crop = frame[y:y+h, x:x+w]
        
        crop_resized = cv2.resize(crop, (64, 128))
        
        filename = os.path.join(self.pos_dir, f"pos_{int(time.time()*1000)}.jpg")
        cv2.imwrite(filename, crop_resized)
        
    def save_negative(self, frame, bbox_to_avoid=None):
        
        if frame is None: return
        h_img, w_img, _ = frame.shape
        
        win_w, win_h = 64, 128
        
        for _ in range(5):
            x = random.randint(0, w_img - win_w)
            y = random.randint(0, h_img - win_h)
            
            if bbox_to_avoid:
                bx, by, bw, bh = bbox_to_avoid
                if not (x > bx + bw or x + win_w < bx or y > by + bh or y + win_h < by):
                    continue # Overlap
            
            crop = frame[y:y+win_h, x:x+win_w]
            filename = os.path.join(self.neg_dir, f"neg_{int(time.time()*1000)}.jpg")
            cv2.imwrite(filename, crop)
            break
            
    def get_sample_count(self):
        pos_count = len(os.listdir(self.pos_dir))
        neg_count = len(os.listdir(self.neg_dir))
        return pos_count + neg_count

    def clear_data(self):
        
        for dr in [self.pos_dir, self.neg_dir]:
            if os.path.exists(dr):
                for f in os.listdir(dr):
                    os.remove(os.path.join(dr, f))
                    
    def get_dataset_size(self):
        
        total_size = 0
        for dr in [self.pos_dir, self.neg_dir]:
            if os.path.exists(dr):
                for f in os.listdir(dr):
                    total_size += os.path.getsize(os.path.join(dr, f))
        return total_size
