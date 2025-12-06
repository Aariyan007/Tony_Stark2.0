# jarvis_camera.py - INTEGRATED WITH YOUR PIPELINE

import cv2
import numpy as np
import time
from datetime import datetime
import threading
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import your existing pipeline
from core.pipeline import process_image
from modules.graph.neo4j_client import Neo4jClient

# Optional voice
try:
    import pyttsx3
    AUDIO_ENABLED = True
except ImportError:
    AUDIO_ENABLED = False
    print("⚠️  Audio disabled (install with: pip install pyttsx3)")


class JarvisCamera:
    """JARVIS Camera integrated with your existing OSINT pipeline"""
    
    def __init__(self):
        print("🔧 Initializing JARVIS systems...")
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Error: Could not open camera")
            sys.exit(1)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Voice
        self.voice_enabled = False
        if AUDIO_ENABLED:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 0.9)
                self.voice_enabled = True
                print("✅ Voice enabled")
            except:
                pass
        
        # State
        self.scanning = False
        self.pipeline_running = False
        self.current_results = None
        self.face_box = None
        self.last_scan_time = 0
        self.scan_cooldown = 3.0
        self.scan_stage = "idle"  # idle, analyzing, complete
        
        # Animation
        self.scan_line_y = 0
        self.scan_direction = 1
        self.frame_count = 0
        self.pulse_alpha = 0
        self.pulse_direction = 1
        self.grid_offset = 0
        
        # Colors (BGR)
        self.COLOR_CYAN = (255, 255, 0)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_RED = (0, 0, 255)
        
        # Face cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # FPS
        self.fps = 0
        self.frame_times = []
        
        print("✅ JARVIS initialized")
    
    def speak(self, text):
        """Text-to-speech"""
        print(f"🗣️  JARVIS: {text}")
        if self.voice_enabled:
            threading.Thread(
                target=lambda: self.engine.say(text) or self.engine.runAndWait(),
                daemon=True
            ).start()
    
    def run(self):
        """Main loop"""
        print("\n" + "="*70)
        print("🤖 JARVIS LIVE CAMERA SYSTEM")
        print("="*70)
        
        self.speak("JARVIS online. Face recognition systems active.")
        
        print("\n⌨️  Controls:")
        print("   SPACE - Run full OSINT pipeline on detected face")
        print("   A     - Auto-scan mode")
        print("   S     - Save screenshot")
        print("   R     - Reset")
        print("   G     - Toggle grid")
        print("   Q     - Quit")
        print("\n✨ Ready to scan!\n")
        
        auto_scan = False
        show_grid = True
        
        while True:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Auto-scan
            if auto_scan and len(faces) > 0 and not self.scanning and not self.pipeline_running:
                current_time = time.time()
                if current_time - self.last_scan_time > self.scan_cooldown:
                    self.trigger_pipeline(frame, faces[0])
            
            # Update animations
            self.update_animations()
            
            # Draw HUD
            frame = self.draw_hud(frame, faces, show_grid)
            
            # Show FPS
            self.draw_fps(frame)
            
            # Display
            cv2.imshow('JARVIS - Face Recognition System', frame)
            
            # FPS calculation
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            if len(self.frame_times) > 0:
                self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
            
            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.speak("Shutting down JARVIS systems")
                break
            elif key == ord(' '):
                if len(faces) > 0 and not self.scanning and not self.pipeline_running:
                    self.trigger_pipeline(frame, faces[0])
            elif key == ord('a'):
                auto_scan = not auto_scan
                mode = "enabled" if auto_scan else "disabled"
                self.speak(f"Auto-scan mode {mode}")
                print(f"🔄 Auto-scan: {mode}")
            elif key == ord('s'):
                self.save_frame(frame)
            elif key == ord('r'):
                self.reset()
            elif key == ord('g'):
                show_grid = not show_grid
        
        self.cleanup()
    
    def detect_faces(self, frame):
        """Detect faces"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        return faces
    
    def trigger_pipeline(self, frame, face):
        """Trigger your full OSINT pipeline"""
        current_time = time.time()
        
        if current_time - self.last_scan_time < self.scan_cooldown:
            return
        
        self.last_scan_time = current_time
        self.scanning = True
        self.pipeline_running = True
        self.face_box = face
        self.scan_stage = "analyzing"
        
        self.speak("Face detected. Initiating full reconnaissance.")
        
        # Extract and save face
        x, y, w, h = face
        pad = 20
        y1 = max(0, y - pad)
        y2 = min(frame.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(frame.shape[1], x + w + pad)
        
        face_img = frame[y1:y2, x1:x2]
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"temp_scan_{timestamp}.jpg"
        cv2.imwrite(temp_path, face_img)
        
        # Run pipeline in thread
        threading.Thread(
            target=self.run_pipeline,
            args=(temp_path,),
            daemon=True
        ).start()
    
    def run_pipeline(self, image_path):
        """Run your full OSINT pipeline"""
        print("\n" + "="*70)
        print("🚀 RUNNING FULL OSINT PIPELINE")
        print("="*70)
        
        self.speak("Running deep analysis")
        
        try:
            # Call YOUR process_image function!
            results = process_image(image_path)
            
            # Store results
            self.current_results = results
            self.scan_stage = "complete"
            
            # Announce completion
            if results.get("instagram"):
                confidence = results["instagram"].get("face_match_result", {}).get("confidence", 0)
                if confidence > 80:
                    self.speak(f"Identity confirmed. High confidence match.")
                else:
                    self.speak(f"Analysis complete. Confidence {confidence:.0f} percent")
            else:
                self.speak("Analysis complete")
            
            print("\n✅ Pipeline completed successfully")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            self.speak("Analysis error occurred")
            import traceback
            traceback.print_exc()
            self.current_results = None
        
        finally:
            self.scanning = False
            self.pipeline_running = False
    
    def update_animations(self):
        """Update animation states"""
        # Pulse
        self.pulse_alpha += self.pulse_direction * 0.02
        if self.pulse_alpha >= 1:
            self.pulse_alpha = 1
            self.pulse_direction = -1
        elif self.pulse_alpha <= 0:
            self.pulse_alpha = 0
            self.pulse_direction = 1
        
        # Grid
        self.grid_offset = (self.grid_offset + 1) % 20
        
        # Scan line
        if self.scanning:
            self.scan_line_y += self.scan_direction * 8
            if self.scan_line_y >= 200:
                self.scan_line_y = 200
                self.scan_direction = -1
            elif self.scan_line_y <= 0:
                self.scan_line_y = 0
                self.scan_direction = 1
    
    def draw_hud(self, frame, faces, show_grid):
        """Draw HUD overlay"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Grid
        if show_grid:
            self.draw_grid(overlay, w, h)
        
        # Corners
        self.draw_corners(overlay, w, h)
        
        # Status bar
        self.draw_status(overlay, w)
        
        # Faces
        for face in faces:
            x, y, fw, fh = face
            self.draw_face_box(overlay, x, y, fw, fh)
            
            # Scan animation
            if self.scanning and self.face_box is not None:
                bx, by, bw, bh = self.face_box
                if x == bx and y == by:
                    self.draw_scan_animation(overlay, x, y, fw, fh)
        
        # Results panel
        if self.current_results:
            self.draw_results_panel(overlay, w, h)
        
        # Pipeline status
        if self.pipeline_running:
            self.draw_pipeline_status(overlay, w, h)
        
        # Instructions
        self.draw_instructions(overlay, w, h)
        
        # Blend
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        return frame
    
    def draw_grid(self, frame, w, h):
        """Draw grid"""
        size = 40
        for i in range(0, w, size):
            x = i + self.grid_offset
            cv2.line(frame, (x, 0), (x, h), self.COLOR_CYAN, 1)
        for i in range(0, h, size):
            y = i + self.grid_offset
            cv2.line(frame, (0, y), (w, y), self.COLOR_CYAN, 1)
    
    def draw_corners(self, frame, w, h):
        """Draw corners"""
        length = 60
        thickness = 3
        intensity = 0.5 + 0.5 * self.pulse_alpha
        color = tuple(int(c * intensity) for c in self.COLOR_CYAN)
        
        # Top-left
        cv2.line(frame, (20, 20), (20 + length, 20), color, thickness)
        cv2.line(frame, (20, 20), (20, 20 + length), color, thickness)
        
        # Top-right
        cv2.line(frame, (w-20, 20), (w-20-length, 20), color, thickness)
        cv2.line(frame, (w-20, 20), (w-20, 20+length), color, thickness)
        
        # Bottom-left
        cv2.line(frame, (20, h-20), (20+length, h-20), color, thickness)
        cv2.line(frame, (20, h-20), (20, h-20-length), color, thickness)
        
        # Bottom-right
        cv2.line(frame, (w-20, h-20), (w-20-length, h-20), color, thickness)
        cv2.line(frame, (w-20, h-20), (w-20, h-20-length), color, thickness)
    
    def draw_status(self, frame, w):
        """Draw status bar"""
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        
        # Title
        cv2.putText(frame, "J.A.R.V.I.S.", 
                   (w//2 - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.2, self.COLOR_CYAN, 3)
        
        cv2.putText(frame, "OSINT RECONNAISSANCE SYSTEM", 
                   (w//2 - 220, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, self.COLOR_WHITE, 1)
        
        # Time
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_str, 
                   (w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, self.COLOR_CYAN, 2)
        
        # Status
        if self.pipeline_running:
            status = "ANALYZING..."
            color = self.COLOR_YELLOW
        elif self.current_results:
            status = "COMPLETE"
            color = self.COLOR_GREEN
        else:
            status = "STANDBY"
            color = self.COLOR_CYAN
        
        cv2.putText(frame, status, 
                   (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2)
    
    def draw_face_box(self, frame, x, y, w, h):
        """Draw face box"""
        corner_len = 25
        thickness = 3
        intensity = 0.5 + 0.5 * self.pulse_alpha
        color = tuple(int(c * intensity) for c in self.COLOR_CYAN)
        
        # Corners
        cv2.line(frame, (x, y), (x+corner_len, y), color, thickness)
        cv2.line(frame, (x, y), (x, y+corner_len), color, thickness)
        
        cv2.line(frame, (x+w, y), (x+w-corner_len, y), color, thickness)
        cv2.line(frame, (x+w, y), (x+w, y+corner_len), color, thickness)
        
        cv2.line(frame, (x, y+h), (x+corner_len, y+h), color, thickness)
        cv2.line(frame, (x, y+h), (x, y+h-corner_len), color, thickness)
        
        cv2.line(frame, (x+w, y+h), (x+w-corner_len, y+h), color, thickness)
        cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_len), color, thickness)
    
    def draw_scan_animation(self, frame, x, y, w, h):
        """Draw scan animation"""
        line_y = y + self.scan_line_y
        
        # Scan lines
        for offset in [-2, 0, 2]:
            alpha = 1.0 - abs(offset) / 4.0
            color = tuple(int(c * alpha) for c in self.COLOR_CYAN)
            cv2.line(frame, (x, line_y + offset), (x+w, line_y + offset), color, 2)
        
        # Glow
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, line_y-15), (x+w, line_y+15), 
                     self.COLOR_CYAN, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Text
        cv2.putText(frame, "SCANNING...", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_CYAN, 2)
    
    def draw_pipeline_status(self, frame, w, h):
        """Draw pipeline status"""
        panel_w = 400
        panel_h = 150
        panel_x = (w - panel_w) // 2
        panel_y = h - panel_h - 80
        
        # Background
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     (10, 10, 10), -1)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     self.COLOR_CYAN, 2)
        
        y_pos = panel_y + 35
        
        # Title
        cv2.putText(frame, "PIPELINE RUNNING", 
                   (panel_x + 80, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_CYAN, 2)
        y_pos += 30
        
        # Stages
        stages = [
            "Face Analysis",
            "Social Media Scan",
            "Database Search",
            "Network Analysis"
        ]
        
        for stage in stages:
            cv2.putText(frame, f"• {stage}", 
                       (panel_x + 20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
            y_pos += 25
    
    def draw_results_panel(self, frame, w, h):
        """Draw results panel with YOUR pipeline results"""
        results = self.current_results
        
        panel_w = 450
        panel_h = min(600, h - 120)
        panel_x = w - panel_w - 30
        panel_y = 90
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     self.COLOR_CYAN, 3)
        
        y_pos = panel_y + 35
        
        # Header
        cv2.putText(frame, "OSINT RESULTS", 
                   (panel_x + 15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_CYAN, 2)
        y_pos += 10
        cv2.line(frame, (panel_x + 15, y_pos), 
                (panel_x + panel_w - 15, y_pos), 
                self.COLOR_CYAN, 2)
        y_pos += 25
        
        # Person ID
        person_id = results.get("person_id", "unknown")
        cv2.putText(frame, f"ID: {person_id}", 
                   (panel_x + 15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_WHITE, 1)
        y_pos += 30
        
        # Face Analysis
        analysis = results.get("analysis", [{}])[0]
        if analysis:
            age = analysis.get("age", "N/A")
            gender = analysis.get("dominant_gender", "N/A")
            cv2.putText(frame, f"Age: {age} | Gender: {gender}", 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
            y_pos += 30
        
        # Instagram
        instagram = results.get("instagram")
        if instagram:
            face_match = instagram.get("face_match_result", {})
            confidence = face_match.get("confidence", 0)
            verified = face_match.get("verified", False)
            
            cv2.putText(frame, "INSTAGRAM:", 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CYAN, 1)
            y_pos += 22
            
            status = "✓ VERIFIED" if verified else "○ Unverified"
            color = self.COLOR_GREEN if verified else self.COLOR_YELLOW
            cv2.putText(frame, f"  {status}", 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 20
            
            cv2.putText(frame, f"  Confidence: {confidence:.1f}%", 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
            y_pos += 25
        
        # GitHub
        github = results.get("github")
        if github:
            cv2.putText(frame, "GITHUB:", 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CYAN, 1)
            y_pos += 22
            
            username = github.get("username", "N/A")
            cv2.putText(frame, f"  @{username}", 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
            y_pos += 20
            
            repos = github.get("public_repos", 0)
            followers = github.get("followers", 0)
            cv2.putText(frame, f"  {repos} repos | {followers} followers", 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
            y_pos += 25
        
        # LinkedIn
        linkedin = results.get("linkedin", {})
        linkedin_scraped = linkedin.get("scraped")
        if linkedin_scraped:
            cv2.putText(frame, "LINKEDIN:", 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CYAN, 1)
            y_pos += 22
            
            name = linkedin_scraped.get("name", "N/A")
            cv2.putText(frame, f"  {name}", 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
            y_pos += 20
            
            headline = linkedin_scraped.get("headline", "")[:40]
            if headline:
                cv2.putText(frame, f"  {headline}", 
                           (panel_x + 15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
                y_pos += 25
        
        # Locations
        locations = results.get("locations", [])
        if locations and y_pos < panel_y + panel_h - 80:
            y_pos += 10
            cv2.putText(frame, "LOCATIONS:", 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CYAN, 1)
            y_pos += 22
            
            for loc in locations[:3]:
                cv2.putText(frame, f"  📍 {loc['name']}", 
                           (panel_x + 15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
                y_pos += 20
        
        # Data completeness
        completeness = results.get("data_completeness", 0)
        if completeness and y_pos < panel_y + panel_h - 50:
            y_pos += 15
            cv2.putText(frame, "DATA QUALITY:", 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_CYAN, 1)
            y_pos += 22
            
            # Bar
            bar_w = panel_w - 50
            bar_h = 18
            bar_x = panel_x + 25
            
            cv2.rectangle(frame, (bar_x, y_pos), 
                         (bar_x + bar_w, y_pos + bar_h), 
                         (40, 40, 40), -1)
            
            fill_w = int(bar_w * completeness / 100)
            bar_color = self.COLOR_GREEN if completeness > 70 else self.COLOR_YELLOW
            cv2.rectangle(frame, (bar_x, y_pos), 
                         (bar_x + fill_w, y_pos + bar_h), 
                         bar_color, -1)
            
            cv2.putText(frame, f"{completeness:.0f}%", 
                       (bar_x + bar_w//2 - 25, y_pos + 14), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 2)
    
    def draw_instructions(self, frame, w, h):
        """Draw instructions"""
        y = h - 25
        instructions = [
            "SPACE:Scan", "A:Auto", "S:Save", "R:Reset", "G:Grid", "Q:Quit"
        ]
        
        x_pos = 40
        for inst in instructions:
            cv2.putText(frame, inst, (x_pos, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_CYAN, 1)
            x_pos += 140
    
    def draw_fps(self, frame):
        """Draw FPS"""
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                   (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, self.COLOR_GREEN, 1)
    
    def save_frame(self, frame):
        """Save screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captures/jarvis_{timestamp}.jpg"
        Path("captures").mkdir(exist_ok=True)
        cv2.imwrite(filename, frame)
        self.speak("Screenshot captured")
        print(f"📸 Saved: {filename}")
    
    def reset(self):
        """Reset"""
        self.current_results = None
        self.scanning = False
        self.pipeline_running = False
        self.scan_stage = "idle"
        self.speak("System reset")
        print("🔄 Reset")
    
    def cleanup(self):
        """Cleanup"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n👋 JARVIS offline")


def main():
    """Run JARVIS"""
    try:
        jarvis = JarvisCamera()
        jarvis.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()