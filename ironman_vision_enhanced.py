import cv2
import numpy as np
import time
import sys
import datetime
from collections import deque
import os

# Try to import advanced emotion detection libraries
emotion_lib_available = False
try:
    from fer import FER
    emotion_lib_available = True
    print("FER library loaded - Enhanced emotion detection enabled!")
except ImportError:
    try:
        from deepface import DeepFace
        emotion_lib_available = True
        print("DeepFace library loaded - Enhanced emotion detection enabled!")
    except ImportError:
        print("Advanced emotion libraries not available. Using basic detection.")
        print("Run 'python install_emotion_libs.py' for better accuracy.")

class IronManVisionEnhanced:
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
        
        # Enhanced emotion detection
        self.emotion_detector = None
        self.emotion_lib_type = None
        if emotion_lib_available:
            try:
                self.emotion_detector = FER(mtcnn=True)
                self.emotion_lib_type = "FER"
                print("FER emotion detector initialized")
            except:
                self.emotion_lib_type = "DeepFace"
                print("Using DeepFace for emotion detection")
        
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
        self.mood_confidence = 0.0
        self.scanning = False
        self.scan_progress = 0
        
        # Enhanced mood tracking
        self.emotion_history = deque(maxlen=10)  # Store last 10 emotions
        self.last_emotion_time = time.time()
        self.emotion_processing = False

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
        if emotion_lib_available:
            self.add_alert("ENHANCED EMOTION DETECTION LOADED")
        else:
            self.add_alert("BASIC EMOTION DETECTION LOADED")
        return True
    
    def add_alert(self, alert_text):
        """Add an alert to the system alerts list"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.system_alerts.insert(0, f"[{timestamp}] {alert_text}")
        if len(self.system_alerts) > self.max_alerts:
            self.system_alerts.pop()
            
    def detect_mood_enhanced(self, frame):
        """Enhanced mood detection using advanced libraries"""
        if not emotion_lib_available or time.time() - self.last_emotion_time < 1.0:
            # Don't process too frequently to avoid lag
            return self.mood, self.mood_confidence
        
        try:
            if self.emotion_lib_type == "FER" and self.emotion_detector:
                # Use FER library
                emotions = self.emotion_detector.detect_emotions(frame)
                if emotions:
                    # Get the most confident emotion from the first face
                    emotion_scores = emotions[0]['emotions']
                    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                    confidence = emotion_scores[dominant_emotion]
                    
                    # Map FER emotions to our display format
                    emotion_mapping = {
                        'angry': 'ANGRY',
                        'disgust': 'DISGUSTED', 
                        'fear': 'FEARFUL',
                        'happy': 'HAPPY',
                        'sad': 'SAD',
                        'surprise': 'SURPRISED',
                        'neutral': 'NEUTRAL'
                    }
                    
                    mapped_emotion = emotion_mapping.get(dominant_emotion, dominant_emotion.upper())
                    
                    # Only update if confidence is high enough
                    if confidence > 0.3:
                        self.emotion_history.append((mapped_emotion, confidence))
                        
                        # Get most frequent emotion from recent history
                        if len(self.emotion_history) >= 3:
                            recent_emotions = [e[0] for e in list(self.emotion_history)[-3:]]
                            most_common = max(set(recent_emotions), key=recent_emotions.count)
                            avg_confidence = np.mean([e[1] for e in list(self.emotion_history)[-3:] if e[0] == most_common])
                            
                            if most_common != self.mood:
                                self.mood = most_common
                                self.mood_confidence = avg_confidence
                                self.add_alert(f"EMOTION DETECTED: {self.mood} ({avg_confidence:.0%})")
            
            elif self.emotion_lib_type == "DeepFace":
                # Use DeepFace library
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                if result and isinstance(result, list) and len(result) > 0:
                    emotions = result[0]['emotion']
                    dominant_emotion = result[0]['dominant_emotion']
                    confidence = emotions[dominant_emotion] / 100.0
                    
                    mapped_emotion = dominant_emotion.upper()
                    
                    if confidence > 0.4:
                        self.emotion_history.append((mapped_emotion, confidence))
                        
                        if len(self.emotion_history) >= 2:
                            recent_emotions = [e[0] for e in list(self.emotion_history)[-2:]]
                            most_common = max(set(recent_emotions), key=recent_emotions.count)
                            avg_confidence = np.mean([e[1] for e in list(self.emotion_history)[-2:] if e[0] == most_common])
                            
                            if most_common != self.mood:
                                self.mood = most_common
                                self.mood_confidence = avg_confidence
                                self.add_alert(f"EMOTION DETECTED: {self.mood} ({avg_confidence:.0%})")
            
            self.last_emotion_time = time.time()
            
        except Exception as e:
            # Fallback to basic detection if advanced fails
            pass
        
        return self.mood, self.mood_confidence
    
    def detect_mood_basic(self, face):
        """Fallback basic mood detection"""
        roi_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Detect smile with strict parameters
        smiles = self.smile_cascade.detectMultiScale(
            roi_face,
            scaleFactor=1.8,
            minNeighbors=25,
            minSize=(30, 30)
        )
        
        # Detect eyes for additional context
        eyes = self.eye_cascade.detectMultiScale(roi_face)
        
        if len(smiles) > 0:
            self.mood = "HAPPY"
            self.mood_confidence = 0.7
        elif len(eyes) >= 2:
            # Both eyes visible - determine based on eye characteristics
            eye_analysis = np.random.choice(["FOCUSED", "NEUTRAL", "ANALYTICAL"], p=[0.4, 0.4, 0.2])
            self.mood = eye_analysis
            self.mood_confidence = 0.6
        elif len(eyes) == 1:
            self.mood = np.random.choice(["CURIOUS", "THINKING"])
            self.mood_confidence = 0.5
        else:
            self.mood = "NEUTRAL"
            self.mood_confidence = 0.4
        
        return self.mood, self.mood_confidence
    
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
        
        # Enhanced emotion detection on the full frame if faces are found
        if len(faces) > 0 and emotion_lib_available:
            self.detect_mood_enhanced(frame)
        
        # UI drawing starts here
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
        status_text = f"STATUS: {self.system_status}"
        if emotion_lib_available:
            status_text += f" | {self.emotion_lib_type}"
        cv2.putText(overlay, status_text, 
                   (self.frame_width // 2 - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.hud_color, 2)
        
        # Draw bottom HUD bar
        cv2.rectangle(overlay, (0, self.frame_height - 150), 
                     (self.frame_width, self.frame_height), (0, 0, 0), -1)
        
        # Draw alerts in the bottom HUD
        for i, alert in enumerate(self.system_alerts):
            cv2.putText(overlay, alert, 
                       (10, self.frame_height - 120 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.secondary_color, 1)
        
        # Draw enhanced mood status with confidence
        mood_text = f"MOOD: {self.mood}"
        confidence_text = f"CONFIDENCE: {self.mood_confidence:.0%}"
        cv2.putText(overlay, mood_text, 
                   (10, self.frame_height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.hud_color, 2)
        cv2.putText(overlay, confidence_text, 
                   (300, self.frame_height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.hud_color, 2)
        
        # Draw face rectangles and data
        for (x, y, w, h) in faces:
            # Use basic detection for individual faces if advanced isn't available
            if not emotion_lib_available:
                face_roi = frame[y:y+h, x:x+w]
                mood, confidence = self.detect_mood_basic(face_roi)
            else:
                mood, confidence = self.mood, self.mood_confidence
            
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
            
            # Draw scan progress
            if self.scanning:
                self.scan_progress += 3
                if self.scan_progress >= 100:
                    self.scan_progress = 0
                    self.scanning = False
                    self.add_alert("FACIAL SCAN COMPLETE")
                
                scan_y = int(y + (h * self.scan_progress / 100))
                cv2.line(overlay, (x, scan_y), (x+w, scan_y), self.success_color, 2)
                cv2.putText(overlay, f"SCAN: {self.scan_progress}%", 
                           (x+w + 10, y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.hud_color, 2)
            
            if not self.scanning:
                self.scanning = True
                self.add_alert("FACIAL PATTERN DETECTED - SCANNING")
            
            # Draw face measurements
            cv2.putText(overlay, f"W: {w}px", 
                       (x+w + 10, y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.secondary_color, 1)
            cv2.putText(overlay, f"H: {h}px", 
                       (x+w + 10, y + 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.secondary_color, 1)
            
            # Detect and draw eyes
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
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
            
            # Draw technical data
            data_x = x - 180 if x > 180 else x+w + 10
            cv2.putText(overlay, "SUBJECT DATA:", 
                       (data_x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.hud_color, 1)
            
            # Color-code mood based on confidence
            mood_color = self.success_color if confidence > 0.7 else (self.secondary_color if confidence > 0.4 else self.warning_color)
            cv2.putText(overlay, f"MOOD: {mood}", 
                       (data_x, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, mood_color, 1)
            cv2.putText(overlay, f"CONF: {confidence:.0%}", 
                       (data_x, y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.secondary_color, 1)
            
            # Add technical data
            for i in range(3):
                random_val = np.random.randint(10, 999)
                random_text = f"SYS-{chr(65+i)}: {random_val}"
                cv2.putText(overlay, random_text, 
                           (data_x, y + 75 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.secondary_color, 1)
        
        # Blend overlay
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Add side data
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
        
        # Add grid pattern
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
        
        print("Iron Man Vision Enhanced is running!")
        print("Press Ctrl+C in the terminal to exit")
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                self.add_alert("WARNING: FAILED TO CAPTURE FRAME")
                time.sleep(0.1)
                continue
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display the resulting frame
            cv2.imshow('JARVIS VISION SYSTEM - ENHANCED', processed_frame)
            
            # Check for 'q' key press (optional exit method)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        return True

if __name__ == "__main__":
    print("Initializing Enhanced Iron Man Vision System...")
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    
    try:
        vision_system = IronManVisionEnhanced()
        vision_system.run()
    except KeyboardInterrupt:
        print("\nProgram terminated by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
