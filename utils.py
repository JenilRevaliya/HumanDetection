import cv2
import mediapipe as mp
import numpy as np
def estimate_distance(bbox_height, frame_height):
    
    if frame_height == 0: return "Unknown"
    ratio = bbox_height / frame_height
    
    if ratio > 0.20: return "Very Near"
    elif ratio > 0.09: return "Near"
    elif ratio > 0.05: return "Medium"
    elif ratio > 0.02: return "Far"
    else: return "Very Far"

def get_distance_color(distance):
    colors = { "Very Near": (0, 0, 255), "Near": (0, 128, 255), "Medium": (0, 255, 255),
               "Far": (255, 0, 0), "Very Far": (128, 0, 0), "Unknown": (128, 128, 128) }
    return colors.get(distance, (255, 255, 255))

def draw_overlay(frame, detection_result, fps, accuracy, distance, phone_results=None, svm_verified=False, face_smoother=None, phone_smoother=None, scale_factor=1.0, eyes_closed=False):
    
    height, width, _ = frame.shape
    
    if phone_results and phone_results.detections:
        for detection in phone_results.detections:
            bbox_norm = detection.bounding_box
            x = int(bbox_norm.origin_x / scale_factor)
            y = int(bbox_norm.origin_y / scale_factor)
            w = int(bbox_norm.width / scale_factor)
            h = int(bbox_norm.height / scale_factor)
            
            if phone_smoother:
                smoothed = phone_smoother.update((x, y, w, h))
                if smoothed:
                    x, y, w, h = smoothed
            
            color = (0, 215, 255) if svm_verified else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            label = f"Mobile: {int(detection.categories[0].score * 100)}%"
            if svm_verified: label += " [V]"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        if phone_smoother: phone_smoother.reset()

    face_bbox = None
    
    if detection_result and detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]
        face_landmarks = landmarks[:11]
        
        x_min, x_max = width, 0
        y_min, y_max = height, 0
        
        for lm in face_landmarks:
            x, y = int(lm.x * width), int(lm.y * height)
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if y < y_min: y_min = y
            if y > y_max: y_max = y
            
        padding = 40
        x_min = max(0, x_min - padding)
        x_max = min(width, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(height, y_max + padding)
        
        w = x_max - x_min
        h = y_max - y_min
        
        if face_smoother:
            smoothed = face_smoother.update((x_min, y_min, w, h))
            if smoothed:
                x_min, y_min, w, h = smoothed
                x_max = x_min + w
                y_max = y_min + h
        
        face_bbox = (x_min, y_min, x_max, y_max)
        
        box_color = (147, 20, 255) if eyes_closed else (0, 255, 0) # BGR for Pink/Purple ish
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
        
        for lm in face_landmarks:
             x, y = int(lm.x * width), int(lm.y * height)
             cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

    else:
        if face_smoother: face_smoother.reset()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), font, 0.7, (0, 255, 0), 2)

    if face_bbox:
        x_min, y_min, x_max, _ = face_bbox
        acc_str = f"{int(accuracy * 100)}%"
        color_dist = get_distance_color(distance)
        
        text_lines = [f"Acc: {acc_str}", f"Dist: {distance}"]
        
        if eyes_closed:
            text_lines.append("Eyes: CLOSED")
        
        text_y = y_min - 10
        for line in reversed(text_lines):
            (w, h), _ = cv2.getTextSize(line, font, 0.6, 2)
            if text_y - h < 0: text_y = y_max + h + 10
            
            bg_color = (0,0,0)
            text_color = (255, 255, 255)
            
            if "Eyes: CLOSED" in line:
                 bg_color = (147, 20, 255) # Pink BG
            elif "Acc" in line:
                 pass # White
            elif "Dist" in line:
                 text_color = color_dist
                 
            cv2.rectangle(frame, (x_min, text_y - h - 5), (x_min + w + 10, text_y + 5), bg_color, -1)
            cv2.putText(frame, line, (x_min + 5, text_y), font, 0.6, text_color, 2)
            text_y -= (h + 10)
    else:
         cv2.putText(frame, "No Face Detected", (10, 60), font, 0.7, (0, 0, 255), 2)
         
    frame = draw_watermark(frame, "github.com/JenilRevaliya")

    return frame

def draw_watermark(frame, text):
    
    h, w = frame.shape[:2]
    
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    color = (255, 255, 255) # White
    
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    cx, cy = w // 2, h // 2
    
    angle = 30 # degrees
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    tx = cx - tw // 2
    ty = cy + th // 2
    
    cv2.putText(overlay, text, (tx, ty), font, font_scale, color, thickness)
    
    rotated_overlay = cv2.warpAffine(overlay, M, (w, h))

    alpha = 0.3 # Transparency of watermark
    
    gray_text = cv2.cvtColor(rotated_overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_text, 10, 255, cv2.THRESH_BINARY)

    cv2.addWeighted(overlay, alpha, frame, 1.0, 0, frame)
    
    return frame

    return frame

