import numpy as np

class BoxSmoother:
    
    def __init__(self, alpha=0.6):
        
        self.alpha = alpha
        self.smoothed_box = None
        
    def update(self, bbox):
        
        if bbox is None:
            self.smoothed_box = None
            return None
            
        current_box = np.array(bbox, dtype=np.float32)
        
        if self.smoothed_box is None:
            self.smoothed_box = current_box
        else:
            self.smoothed_box = self.alpha * current_box + (1 - self.alpha) * self.smoothed_box
            
        return tuple(self.smoothed_box.astype(int))
    
    def reset(self):
        self.smoothed_box = None
