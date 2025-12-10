# pluto_camera.py - ENHANCED UI VERSION

import cv2
import numpy as np
import time
from datetime import datetime
import threading
from pathlib import Path
import sys
import math

sys.path.append(str(Path(__file__).parent))

from core.pipeline import process_image
from modules.graph.neo4j_client import Neo4jClient

# Optional voice
try:
    import pyttsx3
    AUDIO_ENABLED = True
except ImportError:
    AUDIO_ENABLED = False


class PlutoCamera:
    """PLUTO - Advanced OSINT Recognition System"""
    
    def __init__(self):
        print("🔧 Initializing PLUTO...")
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Camera error")
            sys.exit(1)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Voice
        self.voice_enabled = False
        if AUDIO_ENABLED:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 160)
                self.voice_enabled = True
            except:
                pass
        
        # State
        self.pipeline_running = False
        self.pipeline_results = None
        self.face_box = None
        self.last_scan_time = 0
        self.scan_cooldown = 5.0
        
        # Animation state
        self.frame_count = 0
        self.pulse_alpha = 0
        self.pulse_direction = 1
        
        # Scanning animations
        self.scan_radius = 0
        self.scan_angle = 0
        self.scan_rings = []
        self.scan_particles = []
        self.hexagon_rotation = 0
        
        # Colors (BGR) - Modern purple/blue theme
        self.COLOR_PRIMARY = (255, 100, 200)      # Bright purple/pink
        self.COLOR_SECONDARY = (255, 200, 100)    # Orange
        self.COLOR_ACCENT = (100, 255, 200)       # Cyan
        self.COLOR_SUCCESS = (100, 255, 100)      # Green
        self.COLOR_WARNING = (100, 200, 255)      # Yellow
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_DARK = (20, 20, 40)
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # FPS
        self.fps = 0
        self.frame_times = []
        
        print("✅ PLUTO ready")
    
    def speak(self, text):
        """Voice output"""
        print(f"🎙️  PLUTO: {text}")
        if self.voice_enabled:
            threading.Thread(
                target=lambda: self.engine.say(text) or self.engine.runAndWait(),
                daemon=True
            ).start()
    
    def run(self):
        """Main loop"""
        print("\n" + "="*70)
        print("🌌 PLUTO - ADVANCED OSINT SYSTEM")
        print("="*70)
        
        self.speak("PLUTO online. Advanced reconnaissance active.")
        
        print("\n⌨️  Controls:")
        print("   SPACE - Run full OSINT scan")
        print("   A     - Auto-scan mode")
        print("   S     - Save screenshot")
        print("   R     - Reset")
        print("   Q     - Quit")
        print("\n✨ System ready!\n")
        
        auto_scan = False
        
        while True:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Auto-scan
            if auto_scan and len(faces) > 0 and not self.pipeline_running:
                current_time = time.time()
                if current_time - self.last_scan_time > self.scan_cooldown:
                    self.trigger_pipeline(frame, faces[0])
            
            # Update animations
            self.update_animations()
            
            # Draw HUD
            frame = self.draw_hud(frame, faces)
            
            # Display
            cv2.imshow('PLUTO - Advanced OSINT System', frame)
            
            # FPS
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            if len(self.frame_times) > 0:
                self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
            
            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.speak("Shutting down PLUTO")
                break
            elif key == ord(' '):
                if len(faces) > 0 and not self.pipeline_running:
                    self.trigger_pipeline(frame, faces[0])
            elif key == ord('a'):
                auto_scan = not auto_scan
                mode = "enabled" if auto_scan else "disabled"
                self.speak(f"Auto-scan {mode}")
            elif key == ord('s'):
                self.save_frame(frame)
            elif key == ord('r'):
                self.reset()
        
        self.cleanup()
    
    def detect_faces(self, frame):
        """Detect faces"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
    
    def trigger_pipeline(self, frame, face):
        """Trigger full pipeline"""
        current_time = time.time()
        
        if current_time - self.last_scan_time < self.scan_cooldown:
            return
        
        self.last_scan_time = current_time
        self.pipeline_running = True
        self.face_box = face
        self.scan_radius = 0
        self.scan_rings = []
        
        self.speak("Initiating deep scan")
        
        # Save face
        x, y, w, h = face
        pad = 20
        y1 = max(0, y - pad)
        y2 = min(frame.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(frame.shape[1], x + w + pad)
        
        face_img = frame[y1:y2, x1:x2]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"temp_scan_{timestamp}.jpg"
        cv2.imwrite(temp_path, face_img)
        
        # Run pipeline
        threading.Thread(
            target=self.run_pipeline,
            args=(temp_path,),
            daemon=True
        ).start()
    
    def run_pipeline(self, image_path):
        """Run full OSINT pipeline"""
        print("\n" + "="*70)
        print("🚀 PLUTO DEEP SCAN INITIATED")
        print("="*70)
        
        try:
            results = process_image(image_path)
            self.pipeline_results = results
            
            self.speak("Analysis complete")
            
            # Announce findings
            if results.get("instagram"):
                confidence = results["instagram"].get("face_match_result", {}).get("confidence", 0)
                if confidence > 80:
                    self.speak(f"Identity verified. {confidence:.0f} percent match")
            
            completeness = results.get("data_completeness", 0)
            if completeness > 0:
                self.speak(f"Data quality {completeness:.0f} percent")
            
            print("\n✅ SCAN COMPLETE")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.speak("Scan error")
            import traceback
            traceback.print_exc()
            self.pipeline_results = None
        
        finally:
            self.pipeline_running = False
    
    def update_animations(self):
        """Update all animations"""
        # Pulse effect
        self.pulse_alpha += self.pulse_direction * 0.03
        if self.pulse_alpha >= 1:
            self.pulse_alpha = 1
            self.pulse_direction = -1
        elif self.pulse_alpha <= 0:
            self.pulse_alpha = 0
            self.pulse_direction = 1
        
        # Scanning animations
        if self.pipeline_running:
            # Expanding radius
            self.scan_radius += 3
            if self.scan_radius > 200:
                self.scan_radius = 0
            
            # Rotating angle
            self.scan_angle += 5
            if self.scan_angle >= 360:
                self.scan_angle = 0
            
            # Add scan rings
            if self.frame_count % 10 == 0:
                self.scan_rings.append({'radius': 0, 'alpha': 1.0})
            
            # Update rings
            for ring in self.scan_rings:
                ring['radius'] += 4
                ring['alpha'] -= 0.02
            
            # Remove faded rings
            self.scan_rings = [r for r in self.scan_rings if r['alpha'] > 0]
            
            # Add particles
            if self.frame_count % 3 == 0:
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(2, 5)
                self.scan_particles.append({
                    'x': 0,
                    'y': 0,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'life': 50
                })
            
            # Update particles
            for p in self.scan_particles:
                p['x'] += p['vx']
                p['y'] += p['vy']
                p['life'] -= 1
            
            # Remove dead particles
            self.scan_particles = [p for p in self.scan_particles if p['life'] > 0]
        
        # Hexagon rotation
        self.hexagon_rotation += 1
        if self.hexagon_rotation >= 360:
            self.hexagon_rotation = 0
    
    def draw_hud(self, frame, faces):
        """Draw enhanced HUD"""
        h, w = frame.shape[:2]
        
        # Create dark overlay for better contrast
        overlay = np.zeros_like(frame)
        overlay[:] = self.COLOR_DARK
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        overlay = frame.copy()
        
        # Draw elements
        self.draw_header(overlay, w)
        self.draw_corner_elements(overlay, w, h)
        
        # Face detection
        for face in faces:
            x, y, fw, fh = face
            self.draw_enhanced_face_box(overlay, x, y, fw, fh)
            
            # Scanning animation
            if self.pipeline_running and self.face_box is not None:
                bx, by, bw, bh = self.face_box
                if x == bx and y == by:
                    self.draw_advanced_scan(overlay, x, y, fw, fh)
        
        # Results panel
        if self.pipeline_results:
            self.draw_results_panel(overlay, w, h)
        
        # Status indicator
        self.draw_status_indicator(overlay, w, h)
        
        # FPS
        self.draw_fps(overlay)
        
        # Blend
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        return frame
    
    def draw_header(self, frame, w):
        """Draw modern header"""
        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 80), (15, 15, 30), -1)
        
        # Accent line
        cv2.rectangle(frame, (0, 78), (w, 80), self.COLOR_PRIMARY, -1)
        
        # Logo area (left)
        logo_x = 30
        logo_y = 40
        
        # Hexagon logo
        hex_size = 25
        points = []
        for i in range(6):
            angle = math.radians(60 * i)
            x = int(logo_x + hex_size * math.cos(angle))
            y = int(logo_y + hex_size * math.sin(angle))
            points.append([x, y])
        points = np.array(points, np.int32)
        
        cv2.polylines(frame, [points], True, self.COLOR_PRIMARY, 2)
        cv2.circle(frame, (logo_x, logo_y), 8, self.COLOR_PRIMARY, -1)
        
        # Title
        cv2.putText(frame, "P L U T O", 
                   (logo_x + 40, logo_y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, self.COLOR_PRIMARY, 2)
        
        cv2.putText(frame, "ADVANCED OSINT SYSTEM", 
                   (logo_x + 40, logo_y + 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, self.COLOR_WHITE, 1)
        
        # Status (right)
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_str, 
                   (w - 150, logo_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, self.COLOR_ACCENT, 2)
        
        # System status
        if self.pipeline_running:
            status = "SCANNING"
            color = self.COLOR_WARNING
        elif self.pipeline_results:
            status = "COMPLETE"
            color = self.COLOR_SUCCESS
        else:
            status = "READY"
            color = self.COLOR_ACCENT
        
        cv2.putText(frame, status, 
                   (w - 150, logo_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 1)
    
    def draw_corner_elements(self, frame, w, h):
        """Draw modern corner elements"""
        length = 40
        thickness = 2
        
        # Pulsing intensity
        intensity = 0.4 + 0.6 * self.pulse_alpha
        color = tuple(int(c * intensity) for c in self.COLOR_PRIMARY)
        
        offset = 20
        
        # Top-left
        cv2.line(frame, (offset, offset), (offset + length, offset), color, thickness)
        cv2.line(frame, (offset, offset), (offset, offset + length), color, thickness)
        cv2.circle(frame, (offset, offset), 3, color, -1)
        
        # Top-right
        cv2.line(frame, (w-offset, offset), (w-offset-length, offset), color, thickness)
        cv2.line(frame, (w-offset, offset), (w-offset, offset+length), color, thickness)
        cv2.circle(frame, (w-offset, offset), 3, color, -1)
        
        # Bottom-left
        cv2.line(frame, (offset, h-offset), (offset+length, h-offset), color, thickness)
        cv2.line(frame, (offset, h-offset), (offset, h-offset-length), color, thickness)
        cv2.circle(frame, (offset, h-offset), 3, color, -1)
        
        # Bottom-right
        cv2.line(frame, (w-offset, h-offset), (w-offset-length, h-offset), color, thickness)
        cv2.line(frame, (w-offset, h-offset), (w-offset, h-offset-length), color, thickness)
        cv2.circle(frame, (w-offset, h-offset), 3, color, -1)
    
    def draw_enhanced_face_box(self, frame, x, y, w, h):
        """Enhanced face detection box"""
        # Corner length
        corner_len = 30
        thickness = 3
        
        # Pulsing color
        intensity = 0.5 + 0.5 * self.pulse_alpha
        color = tuple(int(c * intensity) for c in self.COLOR_ACCENT)
        
        # Draw corners with rounded effect
        # Top-left
        cv2.line(frame, (x, y+10), (x, y), color, thickness)
        cv2.line(frame, (x, y), (x+10, y), color, thickness)
        cv2.line(frame, (x+10, y), (x+corner_len, y), color, thickness)
        cv2.line(frame, (x, y+10), (x, y+corner_len), color, thickness)
        
        # Top-right
        cv2.line(frame, (x+w, y+10), (x+w, y), color, thickness)
        cv2.line(frame, (x+w, y), (x+w-10, y), color, thickness)
        cv2.line(frame, (x+w-10, y), (x+w-corner_len, y), color, thickness)
        cv2.line(frame, (x+w, y+10), (x+w, y+corner_len), color, thickness)
        
        # Bottom-left
        cv2.line(frame, (x, y+h-10), (x, y+h), color, thickness)
        cv2.line(frame, (x, y+h), (x+10, y+h), color, thickness)
        cv2.line(frame, (x+10, y+h), (x+corner_len, y+h), color, thickness)
        cv2.line(frame, (x, y+h-10), (x, y+h-corner_len), color, thickness)
        
        # Bottom-right
        cv2.line(frame, (x+w, y+h-10), (x+w, y+h), color, thickness)
        cv2.line(frame, (x+w, y+h), (x+w-10, y+h), color, thickness)
        cv2.line(frame, (x+w-10, y+h), (x+w-corner_len, y+h), color, thickness)
        cv2.line(frame, (x+w, y+h-10), (x+w, y+h-corner_len), color, thickness)
        
        # Corner dots
        cv2.circle(frame, (x, y), 4, color, -1)
        cv2.circle(frame, (x+w, y), 4, color, -1)
        cv2.circle(frame, (x, y+h), 4, color, -1)
        cv2.circle(frame, (x+w, y+h), 4, color, -1)
    
    def draw_advanced_scan(self, frame, x, y, w, h):
        """Advanced scanning animation"""
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Expanding rings
        for ring in self.scan_rings:
            radius = int(ring['radius'])
            alpha = ring['alpha']
            color = tuple(int(c * alpha) for c in self.COLOR_PRIMARY)
            cv2.circle(frame, (center_x, center_y), radius, color, 2)
        
        # Rotating scan line
        angle_rad = math.radians(self.scan_angle)
        end_x = int(center_x + self.scan_radius * math.cos(angle_rad))
        end_y = int(center_y + self.scan_radius * math.sin(angle_rad))
        cv2.line(frame, (center_x, center_y), (end_x, end_y), self.COLOR_PRIMARY, 2)
        
        # Particles
        for particle in self.scan_particles:
            px = int(center_x + particle['x'])
            py = int(center_y + particle['y'])
            alpha = particle['life'] / 50.0
            color = tuple(int(c * alpha) for c in self.COLOR_ACCENT)
            cv2.circle(frame, (px, py), 2, color, -1)
        
        # Hexagon at center
        hex_size = 15 + int(10 * math.sin(math.radians(self.hexagon_rotation * 2)))
        points = []
        for i in range(6):
            angle = math.radians(60 * i + self.hexagon_rotation)
            px = int(center_x + hex_size * math.cos(angle))
            py = int(center_y + hex_size * math.sin(angle))
            points.append([px, py])
        points = np.array(points, np.int32)
        cv2.polylines(frame, [points], True, self.COLOR_PRIMARY, 2)
        
        # Scan text
        cv2.putText(frame, "DEEP SCAN IN PROGRESS", 
                   (x, y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, self.COLOR_PRIMARY, 2)
        
        # Progress bar
        bar_w = w
        bar_h = 6
        bar_x = x
        bar_y = y - 30
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                     (40, 40, 60), -1)
        
        progress = (self.scan_angle / 360.0)
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), 
                     self.COLOR_PRIMARY, -1)
    
    def draw_results_panel(self, frame, w, h):
        """Modern results panel"""
        results = self.pipeline_results
        
        panel_w = 450
        panel_h = min(600, h - 120)
        panel_x = w - panel_w - 25
        panel_y = 100
        
        # Background with blur effect
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     (20, 20, 40), -1)
        
        # Border with gradient effect
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     self.COLOR_PRIMARY, 2)
        
        # Accent top line
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + 3), 
                     self.COLOR_PRIMARY, -1)
        
        y_pos = panel_y + 40
        
        # Header
        cv2.putText(frame, "SCAN RESULTS", 
                   (panel_x + 20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, self.COLOR_PRIMARY, 2)
        y_pos += 35
        
        # Person ID
        person_id = results.get("person_id", "unknown")
        cv2.putText(frame, f"ID: {person_id.upper()}", 
                   (panel_x + 20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, self.COLOR_ACCENT, 1)
        y_pos += 30
        
        # Face Analysis
        analysis = results.get("analysis", [{}])[0]
        if analysis:
            age = analysis.get("age", "N/A")
            gender = analysis.get("dominant_gender", "N/A")
            
            cv2.putText(frame, f"AGE: {age}  |  GENDER: {gender.upper()}", 
                       (panel_x + 20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45, self.COLOR_WHITE, 1)
            y_pos += 30
        
        # Divider
        cv2.line(frame, (panel_x + 20, y_pos), 
                (panel_x + panel_w - 20, y_pos), 
                self.COLOR_PRIMARY, 1)
        y_pos += 20
        
        # Instagram
        instagram = results.get("instagram")
        if instagram:
            face_match = instagram.get("face_match_result", {})
            confidence = face_match.get("confidence", 0)
            verified = face_match.get("verified", False)
            
            cv2.putText(frame, "INSTAGRAM", 
                       (panel_x + 20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, self.COLOR_ACCENT, 1)
            y_pos += 25
            
            status = "VERIFIED" if verified else "UNVERIFIED"
            color = self.COLOR_SUCCESS if verified else self.COLOR_WARNING
            cv2.putText(frame, f"Status: {status}", 
                       (panel_x + 25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45, color, 1)
            y_pos += 20
            
            # Confidence bar
            bar_w = panel_w - 50
            bar_h = 8
            bar_x = panel_x + 25
            
            cv2.rectangle(frame, (bar_x, y_pos), 
                         (bar_x + bar_w, y_pos + bar_h), 
                         (40, 40, 60), -1)
            
            fill_w = int(bar_w * confidence / 100)
            bar_color = self.COLOR_SUCCESS if confidence > 80 else self.COLOR_WARNING
            cv2.rectangle(frame, (bar_x, y_pos), 
                         (bar_x + fill_w, y_pos + bar_h), 
                         bar_color, -1)
            
            cv2.putText(frame, f"{confidence:.1f}%", 
                       (bar_x + bar_w + 10, y_pos + 7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, self.COLOR_WHITE, 1)
            y_pos += 25
            
            username = instagram.get("username", "N/A")
            cv2.putText(frame, f"@{username}", 
                       (panel_x + 25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45, self.COLOR_WHITE, 1)
            y_pos += 30
        
        # GitHub
        github = results.get("github")
        if github:
            cv2.putText(frame, "GITHUB", 
                       (panel_x + 20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, self.COLOR_ACCENT, 1)
            y_pos += 25
            
            username = github.get("username", "N/A")
            repos = github.get("public_repos", 0)
            followers = github.get("followers", 0)
            
            cv2.putText(frame, f"@{username}", 
                       (panel_x + 25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45, self.COLOR_WHITE, 1)
            y_pos += 20
            
            cv2.putText(frame, f"{repos} repos  |  {followers} followers", 
                       (panel_x + 25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, self.COLOR_WHITE, 1)
            y_pos += 30
        
        # LinkedIn
        linkedin = results.get("linkedin", {})
        linkedin_scraped = linkedin.get("scraped")
        if linkedin_scraped and y_pos < panel_y + panel_h - 100:
            cv2.putText(frame, "LINKEDIN", 
                       (panel_x + 20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, self.COLOR_ACCENT, 1)
            y_pos += 25
            
            name = linkedin_scraped.get("name", "N/A")
            headline = linkedin_scraped.get("headline", "")[:40]
            
            cv2.putText(frame, name, 
                       (panel_x + 25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45, self.COLOR_WHITE, 1)
            y_pos += 20
            
            if headline:
                cv2.putText(frame, headline, 
                           (panel_x + 25, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.35, self.COLOR_WHITE, 1)
                y_pos += 30
        
        # Data Quality
        completeness = results.get("data_completeness", 0)
        if completeness and y_pos < panel_y + panel_h - 60:
            y_pos += 10
            cv2.line(frame, (panel_x + 20, y_pos), 
                    (panel_x + panel_w - 20, y_pos), 
                    self.COLOR_PRIMARY, 1)
            y_pos += 25
            
            cv2.putText(frame, "DATA QUALITY", 
                       (panel_x + 20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, self.COLOR_ACCENT, 1)
            y_pos += 25
            
            bar_w = panel_w - 50
            bar_h = 12
            bar_x = panel_x + 25
            
            cv2.rectangle(frame, (bar_x, y_pos), 
                         (bar_x + bar_w, y_pos + bar_h), 
                         (40, 40, 60), -1)
            
            fill_w = int(bar_w * completeness / 100)
            bar_color = self.COLOR_SUCCESS if completeness > 70 else self.COLOR_WARNING
            cv2.rectangle(frame, (bar_x, y_pos), 
                         (bar_x + fill_w, y_pos + bar_h), 
                         bar_color, -1)
            
            cv2.putText(frame, f"{completeness:.0f}%", 
                       (bar_x + bar_w//2 - 20, y_pos + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, self.COLOR_WHITE, 1)
    
    def draw_status_indicator(self, frame, w, h):
        """Bottom status bar"""
        bar_h = 50
        bar_y = h - bar_h
        
        cv2.rectangle(frame, (0, bar_y), (w, h), (15, 15, 30), -1)
        cv2.rectangle(frame, (0, bar_y), (w, bar_y + 2), self.COLOR_PRIMARY, -1)
        
        # Controls
        controls = [
            ("SPACE", "Deep Scan"),
            ("A", "Auto"),
            ("S", "Save"),
            ("R", "Reset"),
            ("Q", "Exit")
        ]
        
        x_offset = 30
        for key, action in controls:
            # Key
            cv2.putText(frame, key, 
                       (x_offset, bar_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, self.COLOR_PRIMARY, 1)
            
            # Action
            cv2.putText(frame, action, 
                       (x_offset, bar_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.3, self.COLOR_WHITE, 1)
            
            x_offset += 150
    
    def draw_fps(self, frame):
        """FPS counter"""
        cv2.putText(frame, f"FPS: {self.fps:.0f}", 
                   (30, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, self.COLOR_ACCENT, 1)
    
    def save_frame(self, frame):
        """Save screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captures/pluto_{timestamp}.jpg"
        Path("captures").mkdir(exist_ok=True)
        cv2.imwrite(filename, frame)
        self.speak("Screenshot saved")
        print(f"📸 {filename}")
    
    def reset(self):
        """Reset system"""
        self.pipeline_results = None
        self.pipeline_running = False
        self.scan_rings = []
        self.scan_particles = []
        self.speak("System reset")
        print("🔄 Reset")
    
    def cleanup(self):
        """Cleanup"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n👋 PLUTO offline")


def main():
    """Run PLUTO"""
    try:
        pluto = PlutoCamera()
        pluto.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()