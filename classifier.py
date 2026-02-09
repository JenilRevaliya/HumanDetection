import cv2
import os
import numpy as np

class PhoneClassifier:
    def __init__(self, model_path="phone_svm_model.xml"):
        self.model_path = model_path
        self.svm = cv2.ml.SVM_create()
        self.hog = cv2.HOGDescriptor() # Default winSize=(64,128)
        self.trained = False
        
        # Load trained model if exists
        if os.path.exists(self.model_path):
            try:
                self.svm = cv2.ml.SVM_load(self.model_path)
                self.trained = True
                print(f"[INFO] SVM model loaded from {self.model_path}")
            except Exception as e:
                print(f"[WARNING] Failed to load model: {e}")

    def train(self, data_dir="data"):
        
        pos_dir = os.path.join(data_dir, "positives")
        neg_dir = os.path.join(data_dir, "negatives")
        
        samples = []
        labels = []
        
        print("[INFO] Loading training data...")
        
        if os.path.exists(pos_dir):
            for filename in os.listdir(pos_dir):
                if not filename.endswith(('.jpg', '.png')): continue
                path = os.path.join(pos_dir, filename)
                img = cv2.imread(path)
                img = cv2.resize(img, (64, 128)) # Ensure HOG winSize
                
                hist = self.hog.compute(img)
                samples.append(hist)
                labels.append(1) # Class 1 = Phone

        if os.path.exists(neg_dir):
            for filename in os.listdir(neg_dir):
                if not filename.endswith(('.jpg', '.png')): continue
                path = os.path.join(neg_dir, filename)
                img = cv2.imread(path)
                img = cv2.resize(img, (64, 128))
                
                hist = self.hog.compute(img)
                samples.append(hist)
                labels.append(0) # Class 0 = Background
                
        if not samples:
            print("[ERROR] No training data found.")
            return False
            
        samples = np.array(samples, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        print(f"[INFO] Training SVM on {len(samples)} samples...")
        
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        
        self.svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
        self.svm.save(self.model_path)
        self.trained = True
        print("[INFO] Training complete. Model saved.")
        return True

    def predict(self, frame, bbox):
        
        if not self.trained:
            return 0.0
            
        x, y, w, h = bbox
        h_img, w_img, _ = frame.shape
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w < 10 or h < 10: return 0.0
        
        crop = frame[y:y+h, x:x+w]
        crop = cv2.resize(crop, (64, 128))
        
        hist = self.hog.compute(crop)
        hist = np.array([hist], dtype=np.float32)
        
        response = self.svm.predict(hist)[1][0][0]
        
        return float(response)
        
    def clear_model(self):
        
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        self.trained = False
        self.svm = cv2.ml.SVM_create() # Reset SVM
            
    def get_model_size(self):
        
        if os.path.exists(self.model_path):
            return os.path.getsize(self.model_path)
        return 0
