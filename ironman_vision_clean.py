import cv2
import numpy as np
import time
import sys
import datetime
from collections import deque
import os

class IronManVisionLite:
    def __init__(self):
        # Camera settings
        self.camera_index = 0
        self.cap = None
        self.frame_width = 1280
        self.frame_height = 720
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # UI settings
        self.hud_color = (0, 219, 255)  # Iron Man orange-gold
        self.secondary_color = (255, 255, 255)  # White
        self.warning_color = (0, 0, 255)  # Red
        self.success_color = (0, 255, 0)  # Green
        
        # FPS calculation
        self.fps_hist = deque(maxlen=30)
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
        # System status
        self.system_status = "INITIALIZING..."
        self.system_alerts = []
        self.max_alerts = 5
        self.mood = "Unknown"
        self.scanning = False
        self.scan_progress = 0
        
        # Detection history
        self.smile_detected = False
        self.smile_count = 0
        self.mood_timer = time.time()

    def initialize(self):
        """Initialize the camera and face detection system"""
        self.system_status = "JARVIS ONLINE"
        self.add_alert("JARVIS VISUAL INTERFACE INITIATED")
        
        # Try to open the camera with DirectShow backend on Windows
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Add a small delay to let the camera initialize
        time.sleep(0.5)
        
        if not self.cap.isOpened():
            self.system_status = "CAMERA FAILURE"
            self.add_alert("ERROR: FAILED TO ACCESS PRIMARY VISUAL SENSORS")
            return False
        
        self.add_alert("CAMERA SYSTEMS ONLINE")
        self.add_alert("FACE DETECTION MODULE LOADED")
        return True
    
    def add_alert(self, alert_text):
        """Add an alert to the system alerts list"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.system_alerts.insert(0, f"[{timestamp}] {alert_text}")
        if len(self.system_alerts) > self.max_alerts:
            self.system_alerts.pop()
            
    def detect_mood(self, face):
        """More accurate mood detection based on smile detection with improved parameters"""
        roi_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Detect smile with more strict parameters
        smiles = self.smile_cascade.detectMultiScale(
            roi_face,
            scaleFactor=1.8,
            minNeighbors=30,  # Increased to make smile detection more strict
            minSize=(30, 30)
        )
        
        if len(smiles) > 0:
            self.smile_count += 1
            if self.smile_count > 10 and not self.smile_detected:  # Need more consistent smiles
                self.smile_detected = True
                self.mood = "HAPPY"
                self.add_alert("MOOD DETECTED: HAPPY")
        else:
            self.smile_count = max(0, self.smile_count - 2)  # Decrease faster
            if self.smile_count == 0 and self.smile_detected:
                self.smile_detected = False
                self.mood = "NEUTRAL"
                self.add_alert("MOOD UPDATED: NEUTRAL")
        
        # Update mood every 3 seconds if no smile is detected
        if time.time() - self.mood_timer > 3 and not self.smile_detected:
            self.mood_timer = time.time()
            
            # Determine mood based on eyes and other factors
            eyes = self.eye_cascade.detectMultiScale(roi_face)
            
            if len(eyes) >= 2:  # Both eyes visible
                moods = ["FOCUSED", "ANALYTICAL", "NEUTRAL"]
                self.mood = np.random.choice(moods, p=[0.5, 0.3, 0.2])  # Higher chance of focused
            elif len(eyes) == 1:  # Only one eye visible - may be looking sideways
                moods = ["CURIOUS", "DISTRACTED", "THINKING"]
                self.mood = np.random.choice(moods)
            else:  # No eyes detected
                moods = ["NEUTRAL", "UNKNOWN"]
                self.mood = np.random.choice(moods)
                
            self.add_alert(f"MOOD UPDATED: {self.mood}")
        
        return self.mood, self.smile_detected
    
    def process_frame(self, frame):
        """Process a single frame with face detection and UI overlay"""
        # Calculate FPS
        self.new_frame_time = time.time()
        fps = 1/(self.new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.new_frame_time
        self.fps_hist.append(fps)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # UI drawing starts here
        # Create an overlay for the HUD
        overlay = frame.copy()
        
        # Draw the top HUD bar
        cv2.rectangle(overlay, (0, 0), (self.frame_width, 50), (0, 0, 0), -1)
        
        # Draw status info with creator credit
        cv2.putText(overlay, f"ENGINEERED BY ABED AL RAHMAN", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.hud_color, 2)
        
        # Draw FPS counter
        avg_fps = sum(self.fps_hist)/len(self.fps_hist) if self.fps_hist else 0
        cv2.putText(overlay, f"FPS: {avg_fps:.1f}", 
                   (self.frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.hud_color, 2)
        
        # Draw system status
        cv2.putText(overlay, f"STATUS: {self.system_status}", 
                   (self.frame_width // 2 - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.hud_color, 2)
        
        # Draw bottom HUD bar
        cv2.rectangle(overlay, (0, self.frame_height - 150), 
                     (self.frame_width, self.frame_height), (0, 0, 0), -1)
        
        # Draw alerts in the bottom HUD
        for i, alert in enumerate(self.system_alerts):
            cv2.putText(overlay, alert, 
                       (10, self.frame_height - 120 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.secondary_color, 1)
        
        # Draw mood status
        mood_text = f"MOOD: {self.mood}"
        cv2.putText(overlay, mood_text, 
                   (10, self.frame_height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.hud_color, 2)
        
        # Draw face rectangles and data
        for (x, y, w, h) in faces:
            # Extract face ROI for mood detection
            face_roi = frame[y:y+h, x:x+w]
            mood, smile_detected = self.detect_mood(face_roi)
            
            # Draw face bounding box
            cv2.rectangle(overlay, (x, y), (x+w, y+h), self.hud_color, 2)
            
            # Draw targeting corners (Iron Man style)
            corner_length = 20
            # Top-left corner
            cv2.line(overlay, (x, y), (x + corner_length, y), self.hud_color, 3)
            cv2.line(overlay, (x, y), (x, y + corner_length), self.hud_color, 3)
            # Top-right corner
            cv2.line(overlay, (x+w, y), (x+w - corner_length, y), self.hud_color, 3)
            cv2.line(overlay, (x+w, y), (x+w, y + corner_length), self.hud_color, 3)
            # Bottom-left corner
            cv2.line(overlay, (x, y+h), (x + corner_length, y+h), self.hud_color, 3)
            cv2.line(overlay, (x, y+h), (x, y+h - corner_length), self.hud_color, 3)
            # Bottom-right corner
            cv2.line(overlay, (x+w, y+h), (x+w - corner_length, y+h), self.hud_color, 3)
            cv2.line(overlay, (x+w, y+h), (x+w, y+h - corner_length), self.hud_color, 3)
            
            # Draw scan progress if scanning
            if self.scanning:
                self.scan_progress += 3
                if self.scan_progress >= 100:
                    self.scan_progress = 0
                    self.scanning = False
                    self.add_alert("FACIAL SCAN COMPLETE")
                
                # Draw scan line
                scan_y = int(y + (h * self.scan_progress / 100))
                cv2.line(overlay, (x, scan_y), (x+w, scan_y), self.success_color, 2)
                
                # Draw scan percentage
                cv2.putText(overlay, f"SCAN: {self.scan_progress}%", 
                           (x+w + 10, y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.hud_color, 2)
            
            # If not scanning, start a scan
            if not self.scanning:
                self.scanning = True
                self.add_alert("FACIAL PATTERN DETECTED - SCANNING")
            
            # Draw face measurements (Iron Man style)
            cv2.putText(overlay, f"W: {w}px", 
                       (x+w + 10, y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.secondary_color, 1)
            cv2.putText(overlay, f"H: {h}px", 
                       (x+w + 10, y + 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.secondary_color, 1)
            
            # Detect eyes
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # Draw eyes
            for (ex, ey, ew, eh) in eyes:
                center = (x + ex + ew//2, y + ey + eh//2)
                radius = int(min(ew, eh) * 0.4)
                cv2.circle(overlay, center, radius, self.hud_color, 2)
                
                # Draw crosshair on eyes
                cv2.line(overlay, 
                        (center[0] - radius, center[1]),
                        (center[0] + radius, center[1]), 
                        self.hud_color, 1)
                cv2.line(overlay, 
                        (center[0], center[1] - radius),
                        (center[0], center[1] + radius), 
                        self.hud_color, 1)
            
            # Draw random technical data (for Iron Man aesthetic)
            data_x = x - 180 if x > 180 else x+w + 10
            cv2.putText(overlay, "SUBJECT DATA:", 
                       (data_x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.hud_color, 1)
            cv2.putText(overlay, f"MOOD: {mood}", 
                       (data_x, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       self.success_color if smile_detected else self.secondary_color, 1)
            cv2.putText(overlay, f"DIST: {(w * h) // 1000}k px²", 
                       (data_x, y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.secondary_color, 1)
            
            # Add random technical numbers for Iron Man feel
            for i in range(3):
                random_val = np.random.randint(10, 999)
                random_text = f"SYS-{chr(65+i)}: {random_val}"
                cv2.putText(overlay, random_text, 
                           (data_x, y + 75 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.secondary_color, 1)
        
        # Blend the overlay with the original frame
        alpha = 0.7  # Transparency factor
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Add random calculations on the side for Iron Man feel
        for i in range(5):
            random_val = np.random.randint(100, 999) / 10.0
            random_text = f"DATA-{i+1}: {random_val}"
            cv2.putText(frame, random_text, 
                       (10, 80 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.secondary_color, 1)
        
        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, 
                   (self.frame_width - 200, self.frame_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.secondary_color, 1)
        
        # Add grid pattern for Iron Man HUD feel
        grid_spacing = 80
        grid_alpha = 0.2
        grid_overlay = frame.copy()
        
        for i in range(0, self.frame_width, grid_spacing):
            cv2.line(grid_overlay, (i, 0), (i, self.frame_height), self.hud_color, 1)
        
        for j in range(0, self.frame_height, grid_spacing):
            cv2.line(grid_overlay, (0, j), (self.frame_width, j), self.hud_color, 1)
        
        frame = cv2.addWeighted(grid_overlay, grid_alpha, frame, 1 - grid_alpha, 0)
        
        return frame
    
    def run(self):
        """Main loop to capture and process frames"""
        if not self.initialize():
            return False
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                self.add_alert("WARNING: FAILED TO CAPTURE FRAME")
                time.sleep(0.1)
                continue
            
            # Process the frame
            processed_frame = self.process_frame(frame)
              # Display the resulting frame
            cv2.imshow('JARVIS VISION SYSTEM', processed_frame)
            
            # Check if window was closed by clicking X button
            if cv2.getWindowProperty('JARVIS VISION SYSTEM', cv2.WND_PROP_VISIBLE) < 1:
                # Window was closed by user clicking X
                self.cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
            
            # Press 'q' to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Clean up
                self.cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)  # Ensure complete exit
        
        # This return will never be reached due to sys.exit() above
        # but keeping it for completeness
        return True

if __name__ == "__main__":
    print("Initializing JARVIS Vision System (Lite Version)...")
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    
    try:
        vision_system = IronManVisionLite()
        vision_system.run()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        sys.exit(0)  # Ensure clean exit on keyboard interrupt
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)  # Exit with error code
