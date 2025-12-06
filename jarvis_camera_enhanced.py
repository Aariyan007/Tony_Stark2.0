# jarvis_camera_enhanced.py - ENHANCED JARVIS WITH AUDIO & EFFECTS

import cv2
import numpy as np
from deepface import DeepFace
import time
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
try:
    import pyttsx3
    AUDIO_ENABLED = True
except ImportError:
    AUDIO_ENABLED = False
    print("⚠️  Audio disabled (install pyttsx3 for voice)")

sys.path.append(str(Path(__file__).parent))
from modules.graph.neo4j_client import Neo4jClient


class JarvisEnhanced:
    """Enhanced JARVIS with voice and advanced effects"""
    
    def __init__(self):
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Database
        self.neo = Neo4jClient()
        
        # Voice
        if AUDIO_ENABLED:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed
            self.engine.setProperty('volume', 0.9)  # Volume
        
        # State
        self.scanning = False
        self.scan_progress = 0
        self.current_person = None
        self.face_box = None
        self.last_scan_time = 0
        self.scan_cooldown = 2.0
        
        # Animation state
        self.scan_line_y = 0
        self.scan_direction = 1
        self.frame_count = 0
        self.pulse_alpha = 0
        self.pulse_direction = 1
        self.grid_offset = 0
        self.particles = []
        
        # Colors (BGR)
        self.COLOR_CYAN = (255, 255, 0)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_ORANGE = (0, 165, 255)
        self.COLOR_PURPLE = (255, 0, 255)
        
        # Load cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Performance metrics
        self.fps = 0
        self.frame_times = []
    
    def speak(self, text):
        """Text-to-speech (non-blocking)"""
        if AUDIO_ENABLED:
            threading.Thread(target=lambda: self.engine.say(text) or self.engine.runAndWait(), 
                           daemon=True).start()
        print(f"🗣️  JARVIS: {text}")
    
    def run(self):
        """Main loop"""
        print("="*70)
        print("🤖 JARVIS ENHANCED LIVE SYSTEM")
        print("="*70)
        print("\n📹 Initializing camera systems...")
        
        self.speak("JARVIS online. Face recognition systems active.")
        
        print("\n⌨️  Controls:")
        print("   SPACE - Trigger face scan")
        print("   A     - Auto-scan mode")
        print("   S     - Save frame")
        print("   R     - Reset")
        print("   G     - Toggle grid")
        print("   P     - Toggle particles")
        print("   Q     - Quit")
        print("\n✨ All systems operational!\n")
        
        auto_scan = False
        show_grid = True
        show_particles = True
        
        while True:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Auto-scan mode
            if auto_scan and faces and not self.scanning:
                current_time = time.time()
                if current_time - self.last_scan_time > self.scan_cooldown:
                    self.trigger_scan(frame, faces[0])
            
            # Update animations
            self.update_animations()
            
            # Draw HUD
            frame = self.draw_enhanced_hud(frame, faces, show_grid, show_particles)
            
            # Show FPS
            self.draw_fps(frame)
            
            # Display
            cv2.imshow('JARVIS - Enhanced Recognition System', frame)
            
            # Calculate FPS
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
            
            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.speak("Shutting down JARVIS systems")
                break
            elif key == ord(' '):
                if len(faces) > 0 and not self.scanning:
                    self.trigger_scan(frame, faces[0])
            elif key == ord('a'):
                auto_scan = not auto_scan
                mode = "enabled" if auto_scan else "disabled"
                self.speak(f"Auto-scan mode {mode}")
                print(f"🔄 Auto-scan: {mode}")
            elif key == ord('s'):
                self.save_frame(frame)
            elif key == ord('r'):
                self.reset_detection()
            elif key == ord('g'):
                show_grid = not show_grid
                print(f"Grid: {'ON' if show_grid else 'OFF'}")
            elif key == ord('p'):
                show_particles = not show_particles
                print(f"Particles: {'ON' if show_particles else 'OFF'}")
        
        self.cleanup()
    
    def detect_faces(self, frame):
        """Detect faces"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        return faces
    
    def trigger_scan(self, frame, face):
        """Trigger scan"""
        current_time = time.time()
        
        if current_time - self.last_scan_time < self.scan_cooldown:
            return
        
        self.last_scan_time = current_time
        self.scanning = True
        self.scan_progress = 0
        self.face_box = face
        
        self.speak("Face detected. Initiating scan.")
        
        # Extract and save face
        x, y, w, h = face
        face_img = frame[y:y+h, x:x+w]
        temp_path = "temp_scan.jpg"
        cv2.imwrite(temp_path, face_img)
        
        # Recognize in thread
        threading.Thread(target=self.perform_recognition, 
                        args=(temp_path,), daemon=True).start()
    
    def perform_recognition(self, image_path):
        """Perform recognition"""
        print("\n🔍 Analyzing face patterns...")
        
        try:
            # Generate embedding
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet512",
                enforce_detection=False,
            )
            
            if not embedding:
                self.speak("Unable to extract facial features")
                self.scanning = False
                return
            
            emb_vector = embedding[0]["embedding"]
            
            self.speak("Searching database")
            print("🔎 Comparing against known identities...")
            
            # Search database
            with self.neo.driver.session() as session:
                query = """
                MATCH (p:Person)-[:HAS_FACE]->(e:FaceEmbedding)
                RETURN p.person_id AS person_id,
                       p.name AS name,
                       e.vector AS vector
                """
                results = session.run(query)
                
                best_match = None
                best_similarity = 0
                
                for record in results:
                    db_vector = json.loads(record["vector"])
                    similarity = self._cosine_similarity(emb_vector, db_vector)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            "person_id": record["person_id"],
                            "name": record["name"],
                            "similarity": similarity
                        }
                
                if best_match and best_similarity > 0.6:
                    name = best_match["name"]
                    percent = best_similarity * 100
                    
                    self.speak(f"Identity confirmed. {name}. Match confidence {percent:.0f} percent")
                    print(f"✅ MATCH: {name} ({percent:.1f}%)")
                    
                    # Get full profile
                    profile = self.neo.get_person_profile(best_match["person_id"])
                    if profile:
                        self.current_person = profile
                        self.current_person["similarity"] = best_similarity
                        
                        # Announce details
                        if profile.get("accounts"):
                            acc_count = len(profile["accounts"])
                            self.speak(f"{acc_count} social media accounts found")
                else:
                    self.speak("No match found in database. Identity unknown.")
                    print("❌ No match found")
                    self.current_person = {
                        "name": "UNKNOWN",
                        "person_id": "unknown",
                        "similarity": 0,
                        "accounts": [],
                        "locations": []
                    }
        
        except Exception as e:
            self.speak("Recognition error occurred")
            print(f"❌ Error: {e}")
            self.current_person = None
        
        finally:
            self.scanning = False
            self.scan_progress = 100
    
    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """Calculate cosine similarity"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (mag1 * mag2) if mag1 and mag2 else 0
    
    def update_animations(self):
        """Update animation states"""
        # Pulse effect
        self.pulse_alpha += self.pulse_direction * 0.02
        if self.pulse_alpha >= 1:
            self.pulse_alpha = 1
            self.pulse_direction = -1
        elif self.pulse_alpha <= 0:
            self.pulse_alpha = 0
            self.pulse_direction = 1
        
        # Grid offset
        self.grid_offset = (self.grid_offset + 1) % 20
        
        # Particles
        if len(self.particles) < 50 and np.random.random() < 0.3:
            self.particles.append({
                'x': np.random.randint(0, 1280),
                'y': np.random.randint(0, 720),
                'vx': np.random.uniform(-1, 1),
                'vy': np.random.uniform(-2, 0),
                'life': 100
            })
        
        # Update particles
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
        
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def draw_enhanced_hud(self, frame, faces, show_grid, show_particles):
        """Draw enhanced HUD"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw grid
        if show_grid:
            self.draw_grid(overlay, w, h)
        
        # Draw particles
        if show_particles:
            self.draw_particles(overlay)
        
        # Corner brackets
        self.draw_corner_brackets(overlay, w, h)
        
        # Status bar
        self.draw_status_bar(overlay, w)
        
        # Draw faces
        for face in faces:
            x, y, fw, fh = face
            self.draw_face_box_enhanced(overlay, x, y, fw, fh)
            
            if self.scanning and self.face_box is not None:
                bx, by, bw, bh = self.face_box
                if x == bx and y == by:
                    self.draw_scanning_animation_enhanced(overlay, x, y, fw, fh)
        
        # Person panel
        if self.current_person:
            self.draw_person_panel_enhanced(overlay, w, h)
        
        # Instructions
        self.draw_instructions(overlay, w, h)
        
        # Blend
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        return frame
    
    def draw_grid(self, frame, w, h):
        """Draw animated grid background"""
        grid_size = 40
        alpha = 0.1
        
        for i in range(0, w, grid_size):
            x = i + self.grid_offset
            cv2.line(frame, (x, 0), (x, h), self.COLOR_CYAN, 1)
        
        for i in range(0, h, grid_size):
            y = i + self.grid_offset
            cv2.line(frame, (0, y), (w, y), self.COLOR_CYAN, 1)
    
    def draw_particles(self, frame):
        """Draw floating particles"""
        for p in self.particles:
            alpha = p['life'] / 100.0
            color = tuple(int(c * alpha) for c in self.COLOR_CYAN)
            cv2.circle(frame, (int(p['x']), int(p['y'])), 2, color, -1)
    
    def draw_corner_brackets(self, frame, w, h):
        """Draw corner brackets"""
        length = 60
        thickness = 3
        
        # Animated pulse
        color = tuple(int(c * (0.5 + 0.5 * self.pulse_alpha)) for c in self.COLOR_CYAN)
        
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
    
    def draw_status_bar(self, frame, w):
        """Draw status bar"""
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        
        # Title with glow
        cv2.putText(frame, "J.A.R.V.I.S.", 
                   (w//2 - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.2, self.COLOR_CYAN, 3)
        
        cv2.putText(frame, "FACE RECOGNITION SYSTEM", 
                   (w//2 - 200, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, self.COLOR_WHITE, 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, 
                   (w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, self.COLOR_CYAN, 2)
        
        # Status
        if self.scanning:
            status = "ANALYZING..."
            color = self.COLOR_YELLOW
        elif self.current_person:
            status = "IDENTIFIED"
            color = self.COLOR_GREEN
        else:
            status = "STANDBY"
            color = self.COLOR_CYAN
        
        cv2.putText(frame, status, 
                   (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2)
    
    def draw_face_box_enhanced(self, frame, x, y, w, h):
        """Enhanced face box"""
        corner_len = 25
        thickness = 3
        
        # Pulsing color
        intensity = 0.5 + 0.5 * self.pulse_alpha
        color = tuple(int(c * intensity) for c in self.COLOR_CYAN)
        
        # Draw corners with glow
        corners = [
            [(x, y), (x+corner_len, y), (x, y+corner_len)],
            [(x+w, y), (x+w-corner_len, y), (x+w, y+corner_len)],
            [(x, y+h), (x+corner_len, y+h), (x, y+h-corner_len)],
            [(x+w, y+h), (x+w-corner_len, y+h), (x+w, y+h-corner_len)]
        ]
        
        for corner in corners:
            cv2.line(frame, corner[0], corner[1], color, thickness)
            cv2.line(frame, corner[0], corner[2], color, thickness)
    
    def draw_scanning_animation_enhanced(self, frame, x, y, w, h):
        """Enhanced scanning animation"""
        # Scan line
        self.scan_line_y += self.scan_direction * 8
        
        if self.scan_line_y >= h:
            self.scan_line_y = h
            self.scan_direction = -1
        elif self.scan_line_y <= 0:
            self.scan_line_y = 0
            self.scan_direction = 1
        
        line_y = y + self.scan_line_y
        
        # Multi-line scan effect
        for offset in [-2, 0, 2]:
            alpha = 1.0 - abs(offset) / 4.0
            color = tuple(int(c * alpha) for c in self.COLOR_CYAN)
            cv2.line(frame, (x, line_y + offset), (x+w, line_y + offset), color, 2)
        
        # Glow
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, line_y-15), (x+w, line_y+15), 
                     self.COLOR_CYAN, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Progress
        progress_y = y - 40
        progress_w = int(w * (self.scan_line_y / h))
        
        cv2.rectangle(frame, (x, progress_y), (x+w, progress_y+15), 
                     (30, 30, 30), -1)
        cv2.rectangle(frame, (x, progress_y), (x+progress_w, progress_y+15), 
                     self.COLOR_CYAN, -1)
        
        # Percentage
        percent = int(100 * self.scan_line_y / h)
        cv2.putText(frame, f"SCANNING: {percent}%", 
                   (x, progress_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_CYAN, 2)
    
    def draw_person_panel_enhanced(self, frame, w, h):
        """Enhanced person info panel"""
        person = self.current_person
        
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
        
        # Animated border
        border_color = tuple(int(c * (0.5 + 0.5 * self.pulse_alpha)) for c in self.COLOR_CYAN)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     border_color, 3)
        
        # Content
        y_pos = panel_y + 35
        
        # Header
        cv2.putText(frame, "IDENTITY PROFILE", 
                   (panel_x + 15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_CYAN, 2)
        
        y_pos += 10
        cv2.line(frame, (panel_x + 15, y_pos), 
                (panel_x + panel_w - 15, y_pos), 
                self.COLOR_CYAN, 2)
        y_pos += 25
        
        # Name
        name = person.get("name", "UNKNOWN")
        cv2.putText(frame, name, 
                   (panel_x + 15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.COLOR_WHITE, 2)
        y_pos += 30
        
        # Match confidence
        similarity = person.get("similarity", 0) * 100
        color = self.COLOR_GREEN if similarity > 80 else self.COLOR_YELLOW
        cv2.putText(frame, f"CONFIDENCE: {similarity:.1f}%", 
                   (panel_x + 15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_pos += 35
        
        # Demographics
        if person.get("age") or person.get("gender"):
            demo = []
            if person.get("age"):
                demo.append(f"Age: {person['age']}")
            if person.get("gender"):
                demo.append(f"Gender: {person['gender']}")
            
            cv2.putText(frame, " | ".join(demo), 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
            y_pos += 30
        
        # Accounts
        accounts = person.get("accounts", [])
        if accounts:
            cv2.putText(frame, "SOCIAL ACCOUNTS:", 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CYAN, 1)
            y_pos += 25
            
            for acc in accounts[:6]:
                platform = acc.get("platform", "").upper()
                username = acc.get("username", "N/A")
                verified = "✓" if acc.get("verified") else "○"
                
                cv2.putText(frame, f"  {verified} {platform}", 
                           (panel_x + 15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
                
                cv2.putText(frame, username[:25], 
                           (panel_x + 120, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_CYAN, 1)
                y_pos += 22
        
        # Locations
        locations = person.get("locations", [])
        if locations and y_pos < panel_y + panel_h - 100:
            y_pos += 15
            cv2.putText(frame, "LOCATIONS:", 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CYAN, 1)
            y_pos += 25
            
            for loc in locations[:3]:
                cv2.putText(frame, f"  📍 {loc}", 
                           (panel_x + 15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
                y_pos += 22
        
        # Data completeness
        if person.get("data_completeness") is not None:
            completeness = person["data_completeness"]
            
            y_pos += 20
            cv2.putText(frame, "DATA QUALITY:", 
                       (panel_x + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_CYAN, 1)
            y_pos += 22
            
            # Progress bar
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
            "SPACE:Scan", "A:Auto", "S:Save", "R:Reset", 
            "G:Grid", "P:Particles", "Q:Quit"
        ]
        
        x_pos = 40
        for inst in instructions:
            cv2.putText(frame, inst, (x_pos, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_CYAN, 1)
            x_pos += 140
    
    def draw_fps(self, frame):
        """Draw FPS counter"""
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                   (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, self.COLOR_GREEN, 1)
    
    def save_frame(self, frame):
        """Save frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captures/jarvis_{timestamp}.jpg"
        Path("captures").mkdir(exist_ok=True)
        cv2.imwrite(filename, frame)
        self.speak("Screenshot captured")
        print(f"📸 Saved: {filename}")
    
    def reset_detection(self):
        """Reset"""
        self.current_person = None
        self.scanning = False
        self.speak("Detection reset")
        print("🔄 Reset")
    
    def cleanup(self):
        """Cleanup"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.neo.close()
        print("\n👋 JARVIS offline")


def main():
    try:
        jarvis = JarvisEnhanced()
        jarvis.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()