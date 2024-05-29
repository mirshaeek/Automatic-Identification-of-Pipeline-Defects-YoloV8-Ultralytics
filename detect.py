import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('D:/yolov8/runs/detect/train126/weights/last.pt')

def get_severity_and_solution(defect_type, area):
    if defect_type == "PL":
        if area < 200:
            return "Low", "Rupture: Routine inspection and minor patching"
        elif 200 <= area < 800:
            return "Medium", "Rupture: Moderate patching and protection measures"
        else:
            return "High", "Rupture: Immediate repair and extensive protective measures"
    elif defect_type == "BX":
        if area < 100:
            return "Low", "Deformation: Routine inspection"
        elif 100 <= area < 500:
            return "Medium", "Deformation: Moderate sealing and reinforcement"
        else:
            return "High", "Deformation: Immediate sealing and structural reinforcement"
    if defect_type == "TJ":
        if area < 200:
            return "Low", "Disjoint: Routine inspection and minor patching"
        elif 200 <= area < 800:
            return "Medium", "Disjoint: Moderate patching and protection measures"
        else:
            return "High", "Disjoint: Immediate repair and extensive protective measures"
    if defect_type == "CK":
        if area < 200:
            return "Low", "Misalignment: Routine inspection and minor patching"
        elif 200 <= area < 800:
            return "Medium", "Misalignment: Moderate patching and protection measures"
        else:
            return "High", "Misalignment: Immediate repair and extensive protective measures"
    if defect_type == "CJ":
        if area < 200:
            return "Low", "Deposition: Routine inspection and minor patching"
        elif 200 <= area < 800:
            return "Medium", "Deposition: Moderate patching and protection measures"
        else:
            return "High", "Deposition: Immediate repair and extensive protective measures"
    if defect_type == "ZAW":
        if area < 200:
            return "Low", "Obstacle: Routine inspection and minor patching"
        elif 200 <= area < 800:
            return "Medium", "Obstacle: Moderate patching and protection measures"
        else:
            return "High", "Obstacle: Immediate repair and extensive protective measures"                
    return "Unknown", "Consult an expert for evaluation"

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                conf = box.conf
                cls = box.cls
                defect_type = model.names[int(cls)]
                
                # Crop the defect area for size detection
                defect_area = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(defect_area, cv2.COLOR_BGR2GRAY)
                
                # Apply Canny edge detection
                edges = cv2.Canny(gray, 50, 150)
                
                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Adjust threshold as needed
                        # Get centroid of contour
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # Calculate severity and solution
                        severity, solution = get_severity_and_solution(defect_type, area)
                        
                        # Draw circle around centroid
                        cv2.circle(defect_area, (cX, cY), 5, (0, 255, 0), -1)
                        
                        # Ensure text is within frame boundaries
                        y1_safe = max(y1, 60)
                        cv2.putText(frame, f'{defect_type}', (x1, y1_safe - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
                        cv2.putText(frame, f'Severity: {severity}', (x1, y1_safe - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
                        cv2.putText(frame, f'Solution: {solution}', (x1, y1_safe - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
                        
                # Draw bounding box around defect
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        out.write(frame)
        cv2.imshow('Defect Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
process_video('D:/yolov8/videos/v1.mp4', 'D:/yolov8/videos/output.avi')
