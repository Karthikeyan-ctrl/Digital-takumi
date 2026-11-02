import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import json
import math
import threading


# MediaPipe classes
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# ====================== CONFIGURATION ======================
CAMERA_RESOLUTION = (1280, 720)  # Higher resolution for better tracking
FPS_TARGET = 30                   # Target FPS for consistent playback
COUNTDOWN_SECONDS = 3             # Countdown before recording starts
MIN_RECORDING_FRAMES = 30         # Minimum frames for valid recording
AUTO_STOP_SECONDS = 30            # Auto-stop after this duration
SHOW_SKELETON = True              # Draw skeleton overlay
SHOW_VELOCITY_VECTORS = True      # Show real-time velocity arrows
QUALITY_CHECK_ENABLED = True      # Check for hand visibility during recording
# ===========================================================

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= 1€ FILTER IMPLEMENTATION =================
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize the 1€ filter."""
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # If data arrives too fast or time goes backwards, return previous value
        if t_e <= 0.0: return self.x_prev

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

class HandStabilizer:
    """Manages 1€ filters for all 21 landmarks of a hand."""
    def __init__(self, min_cutoff=0.05, beta=20.0, d_cutoff=1.0): 
        # Low min_cutoff = very smooth when still. High beta = low lag when moving.
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.filters = {} # Stores filters for 'Left' and 'Right'

    def reset(self):
        self.filters = {}

    def update(self, handedness, landmarks, timestamp_sec):
        """
        handedness: 'Left' or 'Right'
        landmarks: List of [x, y, z]
        timestamp_sec: Current time in seconds
        Returns: Stabilized list of [x, y, z]
        """
        # Initialize filters for this hand if not present
        if handedness not in self.filters:
            self.filters[handedness] = []
            for lm in landmarks:
                # 3 filters per landmark (x, y, z)
                f_x = OneEuroFilter(timestamp_sec, lm[0], min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
                f_y = OneEuroFilter(timestamp_sec, lm[1], min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
                f_z = OneEuroFilter(timestamp_sec, lm[2], min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
                self.filters[handedness].append((f_x, f_y, f_z))
            return landmarks # Return raw on first frame

        # Apply filters
        stabilized_landmarks = []
        
        # Check if time jump is too large (hand lost tracking for a while), if so, reset
        if len(self.filters[handedness]) > 0:
             dt = timestamp_sec - self.filters[handedness][0][0].t_prev
             if dt > 1.0: # If lost for > 1 sec, hard reset to avoid "flying" hands
                 del self.filters[handedness]
                 return self.update(handedness, landmarks, timestamp_sec)

        for i, lm in enumerate(landmarks):
            f_x, f_y, f_z = self.filters[handedness][i]
            new_x = f_x.filter(timestamp_sec, lm[0])
            new_y = f_y.filter(timestamp_sec, lm[1])
            new_z = f_z.filter(timestamp_sec, lm[2])
            stabilized_landmarks.append([new_x, new_y, new_z])
            
        return stabilized_landmarks
# ============================================================

# ================= KINEMATIC ENGINE =================
class SkeletalDigitizer:
    def __init__(self):
        # Finger joint indices (MCP, PIP, DIP, TIP)
        # Thumb is exception: CMC, MCP, IP, TIP
        self.fingers_indices = {
            "Thumb": [1, 2, 3, 4],
            "Index": [5, 6, 7, 8],
            "Middle": [9, 10, 11, 12],
            "Ring": [13, 14, 15, 16],
            "Pinky": [17, 18, 19, 20]
        }
        self.WRIST = 0
        self.MIDDLE_MCP = 9  # Middle finger MCP joint index

    def get_vector(self, p1, p2):
        """Calculate vector from point p1 to p2."""
        return np.array(p2) - np.array(p1)

    def normalize(self, v):
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(v)
        if norm == 0: return v
        return v / norm

    def calculate_angle(self, v1, v2):
        """
        Calculate the angle (degrees) between two vectors.
        Used for flexion (bending) calculation.
        """
        v1_u = self.normalize(v1)
        v2_u = self.normalize(v2)
        
        # Dot product = cos(theta)
        dot_prod = np.dot(v1_u, v2_u)
        # Clip to prevent numerical errors outside valid range
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        
        angle_rad = np.arccos(dot_prod)
        return np.degrees(angle_rad)

    def compute_hand_angles(self, landmarks):
        """
        Converts 21 landmarks into a dictionary of joint angles.
        Returns: { 'Index_MCP': 45.0, 'Index_PIP': 88.0, ... }
        """
        angles = {}
        
        # 1. Wrist/Palm Reference
        # Calculate palm normal or global orientation if needed later
        # For now, we focus on local relative angles (Finger Bending)
        
        for name, indices in self.fingers_indices.items():
            # Unpack joint indices
            idx_0, idx_1, idx_2, idx_3 = indices
            
            # --- Vectors representing bones ---
            # Metacarpal (Palm bone) -> Proximal (Finger base)
            # Note: For thumb, this is Wrist->CMC. For others, Wrist->MCP
            if name == "Thumb":
                bone_base = self.get_vector(landmarks[self.WRIST], landmarks[idx_0])
            else:
                # Use Wrist -> MCP as the base reference for the finger
                bone_base = self.get_vector(landmarks[self.WRIST], landmarks[idx_0])

            bone_proximal = self.get_vector(landmarks[idx_0], landmarks[idx_1]) # MCP to PIP
            bone_intermediate = self.get_vector(landmarks[idx_1], landmarks[idx_2]) # PIP to DIP
            bone_distal = self.get_vector(landmarks[idx_2], landmarks[idx_3]) # DIP to Tip

            # --- Calculate Bends (Flexion) ---
            # Angle between Metacarpal and Proximal (Knuckle bend)
            angles[f"{name}_MCP_Flexion"] = self.calculate_angle(bone_base, bone_proximal)
            
            # Angle between Proximal and Intermediate (Middle joint bend)
            angles[f"{name}_PIP_Flexion"] = self.calculate_angle(bone_proximal, bone_intermediate)
            
            # Angle between Intermediate and Distal (Tip joint bend)
            # Thumb only has IP, not DIP, but logic holds (IP is the last joint)
            label_distal = "IP" if name == "Thumb" else "DIP"
            angles[f"{name}_{label_distal}_Flexion"] = self.calculate_angle(bone_intermediate, bone_distal)

        # 2. Abduction (Finger Spread)
        # Angle between adjacent MCP bones (e.g., Index MCP vs Middle MCP)
        # This measures how wide you are spreading your fingers
        mcp_indices = [5, 9, 13, 17] # Index, Middle, Ring, Pinky
        spread_names = ["Index-Mid", "Mid-Ring", "Ring-Pinky"]
        
        for i in range(len(spread_names)):
            idx_a = mcp_indices[i]
            idx_b = mcp_indices[i+1]
            
            vec_a = self.get_vector(landmarks[self.WRIST], landmarks[idx_a])
            vec_b = self.get_vector(landmarks[self.WRIST], landmarks[idx_b])
            
            angles[f"Spread_{spread_names[i]}"] = self.calculate_angle(vec_a, vec_b)

        return angles
    
    def get_spatial_normalized_landmarks(self, landmarks):
        """
        Centers the hand at the wrist and scales it to a unit size.
        Returns normalized landmarks as a list.
        """
        lm_array = np.array(landmarks)
        
        # 1. Translation: Shift everything so Wrist (idx 0) is at 0,0,0
        wrist_pos = lm_array[self.WRIST]
        centered = lm_array - wrist_pos
        
        # 2. Scale: Normalize size based on the "Palm Bone" (Wrist to Middle MCP)
        palm_size = np.linalg.norm(centered[self.MIDDLE_MCP])
        
        if palm_size == 0: 
            return landmarks # Prevent divide by zero error
            
        normalized = centered / palm_size
        
        return normalized.tolist() # Convert back to list for saving
    
    def compute_wrist_orientation(self, landmarks):
        """
        Calculates the global orientation of the hand (Roll, Pitch, Yaw).
        Returns a dictionary of angles in degrees.
        """
        # 1. Define the Plane of the Palm
        # We use 3 points: Wrist(0), Index_MCP(5), Pinky_MCP(17)
        wrist = np.array(landmarks[0])
        index_mcp = np.array(landmarks[5])
        pinky_mcp = np.array(landmarks[17])
        
        # 2. Construct the Local Coordinate System (Basis Vectors)
        
        # Y-axis: Vector pointing from Wrist to Middle Finger (Longitudinal axis)
        # (Using Middle MCP(9) is more stable than Index for center axis)
        middle_mcp = np.array(landmarks[9])
        y_vec = middle_mcp - wrist
        y_vec /= np.linalg.norm(y_vec)
        
        # Z-axis: Normal vector sticking out of the palm
        # Cross product of (Wrist->Index) and (Wrist->Pinky)
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        z_vec = np.cross(v1, v2) # Direction: Out of palm
        z_vec /= np.linalg.norm(z_vec)
        
        # X-axis: Perpendicular to Y and Z (Lateral axis)
        x_vec = np.cross(y_vec, z_vec)
        x_vec /= np.linalg.norm(x_vec)
        
        # 3. Calculate Euler Angles (Roll, Pitch, Yaw)
        # We derive these from the Basis Vectors relative to the Camera (World)
        # Note: This assumes Camera Y is "Up" and Camera Z is "Forward"
        
        # PITCH (Tilting fingers up/down)
        # Angle between Y-vec and World Y-axis
        pitch = np.degrees(np.arcsin(y_vec[1])) 
        
        # YAW (Turning palm left/right)
        # Angle of the Y-vector projected on the XZ plane
        yaw = np.degrees(np.arctan2(y_vec[0], y_vec[2]))
        
        # ROLL (Rotating palm like a doorknob)
        # Angle of the Z-vector (Palm Normal) around the Y-axis
        # We project Z onto the Global X axis to see rotation
        roll = np.degrees(np.arctan2(x_vec[1], z_vec[1]))

        return {
            "wrist_roll": roll,
            "wrist_pitch": pitch,
            "wrist_yaw": yaw
        }

# Recording modes
class RecordingMode:
    IDLE = 0
    COUNTDOWN = 1
    RECORDING = 2
    PAUSED = 3
    REVIEWING = 4

# Statistics tracker
class RecordingStats:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_frames = 0
        self.frames_with_both_hands = 0
        self.frames_with_one_hand = 0
        self.frames_with_no_hands = 0
        self.avg_confidence = 0.0
        self.confidence_sum = 0.0
        self.start_time = None
        self.end_time = None
        self.velocities = []
    
    def update(self, hands_count, confidence=0.0):
        self.total_frames += 1
        if hands_count == 2:
            self.frames_with_both_hands += 1
        elif hands_count == 1:
            self.frames_with_one_hand += 1
        else:
            self.frames_with_no_hands += 1
        
        if confidence > 0:
            self.confidence_sum += confidence
            self.avg_confidence = self.confidence_sum / self.total_frames
    
    def get_duration(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def get_quality_score(self):
        if self.total_frames == 0:
            return 0.0
        # Quality = percentage of frames with both hands visible
        return (self.frames_with_both_hands / self.total_frames) * 100

def draw_hand_enhanced(frame, landmarks, color, thickness=2, show_joints=True, show_labels=False):
    """Enhanced hand drawing with optional joint numbers"""
    h, w, _ = frame.shape
    pts = []
    for lm in landmarks:
        x_px = int(lm[0] * w)
        y_px = int(lm[1] * h)
        pts.append((x_px, y_px))
    
    # Draw connections
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, thickness)
    
    # Draw joints
    if show_joints:
        for i, pt in enumerate(pts):
            cv2.circle(frame, pt, 5, color, -1)
            if show_labels and i in [0, 4, 8, 12, 16, 20]:  # Key joints only
                cv2.putText(frame, str(i), (pt[0]+8, pt[1]-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

def draw_biometric_hud(frame, angles, handedness):
    """
    Draws a sci-fi style dashboard showing finger angles.
    """
    h, w, _ = frame.shape
    
    # --- UPDATE THIS: INCREASE PANEL HEIGHT ---
    panel_w = 250
    panel_h = 350  # Changed from 230 to 350 to fit the Wrist Data
    # ------------------------------------------
    
    # Position: Right side for Right hand, Left side for Left hand
    x_start = w - panel_w - 20 if handedness == "Right" else 20
    y_start = h // 2 - 150 # Moved up slightly to fit taller panel
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Header
    color = (0, 255, 0) if handedness == "Right" else (255, 165, 0)
    cv2.rectangle(frame, (x_start, y_start), (x_start + panel_w, y_start + 5), color, -1)
    cv2.putText(frame, f"{handedness.upper()} HAND BIO-DATA", (x_start + 10, y_start + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Metrics to display (Key joints for "Muscle Memory")
    metrics = [
        ("Thumb MCP", angles.get("Thumb_MCP_Flexion", 0), 60),  
        ("Index PIP", angles.get("Index_PIP_Flexion", 0), 110), 
        ("Mid PIP",   angles.get("Middle_PIP_Flexion", 0), 110),
        ("Ring PIP",  angles.get("Ring_PIP_Flexion", 0), 110),
        ("Pinky PIP", angles.get("Pinky_PIP_Flexion", 0), 110)
    ]
    
    y_off = 50
    for name, value, max_val in metrics:
        # Normalize value for progress bar (0 to 1)
        ratio = min(max(value / max_val, 0), 1)
        bar_w = 100
        filled_w = int(bar_w * ratio)
        
        # Text label
        cv2.putText(frame, f"{name}", (x_start + 10, y_start + y_off), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Numeric value
        cv2.putText(frame, f"{int(value)}deg", (x_start + panel_w - 60, y_start + y_off), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw Bar Background
        bar_x = x_start + 80
        bar_y = y_start + y_off - 5
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6), (50, 50, 50), -1)
        
        # Draw Active Bar (Color changes with intensity)
        bar_color = (0, 255, 255) # Yellow
        if ratio > 0.8: bar_color = (0, 0, 255) # Red (Max flex)
        elif ratio < 0.2: bar_color = (0, 255, 0) # Green (Relaxed)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + 6), bar_color, -1)
        
        y_off += 30

    # Spread Metric (Index vs Pinky width)
    spread = angles.get("Spread_Index-Mid", 0) + angles.get("Spread_Mid-Ring", 0) + angles.get("Spread_Ring-Pinky", 0)
    cv2.putText(frame, f"Total Spread: {int(spread)}", (x_start + 10, y_start + y_off + 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
               
    # --- INSERTED CODE START ---
    # WRIST ORIENTATION SECTION
    cv2.putText(frame, "WRIST ROTATION", (x_start + 10, y_start + y_off + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Helper to draw orientation lines
    def draw_param(label, val, y):
        cv2.putText(frame, f"{label}: {int(val)}", (x_start + 10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    r = angles.get("wrist_roll", 0)
    p = angles.get("wrist_pitch", 0)
    y_val = angles.get("wrist_yaw", 0)

    draw_param(f"Roll  (Tilt)", r, y_start + y_off + 50)
    draw_param(f"Pitch (Up/Dn)", p, y_start + y_off + 70)
    draw_param(f"Yaw   (Turn)", y_val, y_start + y_off + 90)
    # --- INSERTED CODE END ---

def draw_velocity_vector(frame, curr_pos, prev_pos, color=(0, 255, 255), scale=10):
    """Draw velocity vector as arrow"""
    h, w, _ = frame.shape
    curr_x, curr_y = int(curr_pos[0] * w), int(curr_pos[1] * h)
    prev_x, prev_y = int(prev_pos[0] * w), int(prev_pos[1] * h)
    
    vel_x = (curr_x - prev_x) * scale
    vel_y = (curr_y - prev_y) * scale
    
    if abs(vel_x) > 2 or abs(vel_y) > 2:  # Only draw if significant movement
        end_x = curr_x + int(vel_x)
        end_y = curr_y + int(vel_y)
        cv2.arrowedLine(frame, (curr_x, curr_y), (end_x, end_y), color, 2, tipLength=0.3)

def draw_info_panel(frame, mode, stats, countdown_val=0, message=""):
    """Draw comprehensive info panel"""
    h, w, _ = frame.shape
    panel_h = 180
    
    # Semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    y_offset = 30
    line_height = 30
    
    # Mode status
    if mode == RecordingMode.IDLE:
        status = "READY"
        color = (0, 255, 0)
    elif mode == RecordingMode.COUNTDOWN:
        status = f"STARTING IN {countdown_val}..."
        color = (0, 255, 255)
    elif mode == RecordingMode.RECORDING:
        status = "⚫ RECORDING"
        color = (0, 0, 255)
    elif mode == RecordingMode.PAUSED:
        status = "⏸ PAUSED"
        color = (255, 165, 0)
    elif mode == RecordingMode.REVIEWING:
        status = "👁 REVIEWING"
        color = (255, 255, 0)
    
    # FONT_HERSHEY_BOLD does not exist in OpenCV Python; use SIMPLEX (increase thickness for bold)
    cv2.putText(frame, status, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    y_offset += line_height
    
    # Recording stats
    if mode == RecordingMode.RECORDING or mode == RecordingMode.PAUSED or mode == RecordingMode.REVIEWING:
        info_text = f"Frames: {stats.total_frames} | Quality: {stats.get_quality_score():.1f}%"
        cv2.putText(frame, info_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        # Hand visibility breakdown
        hands_text = f"Both Hands: {stats.frames_with_both_hands} | One: {stats.frames_with_one_hand} | None: {stats.frames_with_no_hands}"
        cv2.putText(frame, hands_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        # Confidence
        if stats.avg_confidence > 0:
            conf_text = f"Avg Confidence: {stats.avg_confidence:.2f}"
            cv2.putText(frame, conf_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += line_height
    
    # Custom message
    if message:
        cv2.putText(frame, message, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Controls at bottom
    y_offset = h - 60
    cv2.putText(frame, "R: Record | P: Pause | Space: Review | S: Save | D: Discard | Q: Quit", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, "V: Toggle Velocity | K: Toggle Skeleton | L: Toggle Labels", 
                (20, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

def save_recording_with_metadata(recorded_frames, stats, filename_base):
    """Save recording with metadata (Thread-Safe Version)"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save frame data
        frames_file = f"{filename_base}_{timestamp}.npy"
        np.save(frames_file, recorded_frames)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'total_frames': stats.total_frames,
            'duration_seconds': stats.get_duration(),
            'fps': stats.total_frames / stats.get_duration() if stats.get_duration() > 0 else 0,
            'quality_score': stats.get_quality_score(),
            'frames_with_both_hands': stats.frames_with_both_hands,
            'frames_with_one_hand': stats.frames_with_one_hand,
            'frames_with_no_hands': stats.frames_with_no_hands,
            'avg_confidence': stats.avg_confidence,
            'resolution': CAMERA_RESOLUTION,
            'recording_mode': 'expert_template'
        }
        
        metadata_file = f"{filename_base}_{timestamp}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Print confirmation (will appear in console whenever it finishes)
        print(f"\n{'='*30}")
        print(f"✅ BACKGROUND SAVE COMPLETE")
        print(f"📁 File: {frames_file}")
        print(f"{'='*30}\n")
        
        return frames_file, metadata_file

    except Exception as e:
        print(f"\n❌ BACKGROUND SAVE FAILED: {e}\n")

def playback_recording(frames, landmarker):
    """Playback recorded frames for review"""
    print("\n>>> REVIEW MODE - Playing back recording...")
    print("Controls: Space=Pause/Resume | Right=+5 frames | Left=-5 frames | Q=Exit review")
    
    frame_idx = 0
    paused = False
    playback_speed = 1
    
    while frame_idx < len(frames):
        # Create blank canvas
        canvas = np.zeros((CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
        
        frame_data = frames[frame_idx]
        
        # Draw hands
        for hand_data in frame_data['hands']:
            lm_list = hand_data['landmarks']
            handedness = hand_data['handedness']
            color = (0, 255, 0) if handedness == "Right" else (255, 100, 0)
            draw_hand_enhanced(canvas, lm_list, color, thickness=3)
        
        # Draw playback info
        cv2.putText(canvas, f"Frame: {frame_idx + 1}/{len(frames)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, "REVIEW MODE" if not paused else "PAUSED", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(canvas, "Space: Pause | Arrows: Skip | Q: Exit", (20, CAMERA_RESOLUTION[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        cv2.imshow('Digital Takumi - Review', canvas)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == 83:  # Right arrow
            frame_idx = min(len(frames) - 1, frame_idx + 5)
        elif key == 81:  # Left arrow
            frame_idx = max(0, frame_idx - 5)
        
        if not paused:
            frame_idx += playback_speed
    
    cv2.destroyWindow('Digital Takumi - Review')

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    
    # State variables
    mode = RecordingMode.IDLE
    recorded_frames = []
    previous_frame_data = None
    stats = RecordingStats()
    countdown_start = None
    message = ""
    message_time = None
    stabilizer = HandStabilizer(min_cutoff=0.01, beta=5.0)
    digitizer = SkeletalDigitizer()
    # Display options
    show_skeleton = SHOW_SKELETON
    show_velocity = SHOW_VELOCITY_VECTORS
    show_labels = False
    
    print("\n" + "="*70)
    print("  DIGITAL TAKUMI - ENHANCED GOLDEN TEMPLATE RECORDER")
    print("="*70)
    print("\n📹 RECORDING CONTROLS:")
    print("  R      - Start recording (with countdown)")
    print("  P      - Pause/Resume recording")
    print("  Space  - Review recorded footage")
    print("  S      - Save recording to file")
    print("  D      - Discard current recording")
    print("  Q      - Quit application")
    print("\n🎨 DISPLAY OPTIONS:")
    print("  V      - Toggle velocity vectors")
    print("  K      - Toggle skeleton display")
    print("  L      - Toggle joint labels")
    print("\n⚙️  SETTINGS:")
    print(f"  Resolution: {CAMERA_RESOLUTION[0]}x{CAMERA_RESOLUTION[1]}")
    print(f"  Target FPS: {FPS_TARGET}")
    print(f"  Countdown: {COUNTDOWN_SECONDS} seconds")
    print(f"  Quality Check: {'Enabled' if QUALITY_CHECK_ENABLED else 'Disabled'}")
    print("="*70 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror for intuitive movement
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        h, w, _ = frame.shape
        
        current_frame_data = {
            'timestamp_ms': timestamp_ms,
            'hands': []
        }
        
        hands_detected = 0
        total_confidence = 0.0
        
        # Process detected hands
        # Process detected hands
        if results.hand_landmarks:
            hands_detected = len(results.hand_landmarks)
            
            for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
                # 1. Get raw label
                raw_label = results.handedness[hand_idx][0].category_name if results.handedness else "Unknown"
                
                # 2. SWAP LABELS (Fix for Mirrored Camera)
                # Since we flipped the frame, Right looks like Left. We must correct it.
                if raw_label == "Left":
                    handedness_label = "Right"
                elif raw_label == "Right":
                    handedness_label = "Left"
                else:
                    handedness_label = raw_label
                
                confidence = results.handedness[hand_idx][0].score if results.handedness else 0.0
                total_confidence += confidence
                
                # 1. Get RAW landmarks
                landmarks = hand_landmarks.landmark if hasattr(hand_landmarks, 'landmark') else hand_landmarks
                raw_lm_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
                
                # 2. APPLY 1€ FILTER (Signal Processing)
                # Convert ms to seconds for the filter
                t_sec = timestamp_ms / 1000.0
                lm_list = stabilizer.update(handedness_label, raw_lm_list, t_sec)
                
                # --- COMPUTE KINEMATIC ANGLES ---
                # This converts XYZ positions into pure joint angles (The "True" Muscle Memory)
                joint_angles = digitizer.compute_hand_angles(lm_list)

                # --- WRIST ORIENTATION (6-DoF) ---
                # Captures "Palm Facing" direction
                wrist_orientation = digitizer.compute_wrist_orientation(lm_list)
                
                # Merge into joint_angles for saving (keep it all in one "physics" packet)
                joint_angles.update(wrist_orientation)
                # --------------------------------------

                norm_lm_list = digitizer.get_spatial_normalized_landmarks(lm_list)

                # --- DRAW BIOMETRIC HUD ---
                # visualize the physics we are recording
                if show_skeleton: # Or make a new toggle key 'B' for Biometrics
                    draw_biometric_hud(frame, joint_angles, handedness_label)

                # 3. Save STABILIZED data (clean muscle memory)
                current_frame_data['hands'].append({
                    'handedness': handedness_label,
                    'landmarks': lm_list, # Now saving clean data
                    'joint_angles': joint_angles,
                    'normalized_landmarks': norm_lm_list,
                    'confidence': confidence
                })
                
                # Draw hand (using clean data looks smoother too)
                if show_skeleton:
                    color = (0, 255, 0) if handedness_label == "Right" else (255, 0, 0)
                    draw_hand_enhanced(frame, lm_list, color, thickness=2, 
                                     show_joints=True, show_labels=show_labels)
                
                # Draw velocity vectors (using clean data prevents jittery arrows)
                if show_velocity and previous_frame_data and mode == RecordingMode.RECORDING:
                    prev_hands = {h['handedness']: h['landmarks'] for h in previous_frame_data['hands']}
                    if handedness_label in prev_hands:
                        prev_wrist = prev_hands[handedness_label][0]
                        curr_wrist = lm_list[0]
                        draw_velocity_vector(frame, curr_wrist, prev_wrist)
        
        # Handle countdown mode
        if mode == RecordingMode.COUNTDOWN:
            elapsed = (datetime.now() - countdown_start).total_seconds()
            countdown_val = COUNTDOWN_SECONDS - int(elapsed)
            
            if countdown_val <= 0:
                mode = RecordingMode.RECORDING
                stats.start_time = datetime.now()
                message = "Recording started!"
                message_time = datetime.now()
            else:
                # Draw large countdown number (FONT_HERSHEY_BOLD not in OpenCV Python)
                cv2.putText(frame, str(countdown_val), (w//2 - 50, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
        
        # Recording mode
        if mode == RecordingMode.RECORDING:
            recorded_frames.append(current_frame_data)
            avg_conf = total_confidence / hands_detected if hands_detected > 0 else 0.0
            stats.update(hands_detected, avg_conf)
            
            # Auto-stop check
            if stats.get_duration() >= AUTO_STOP_SECONDS:
                mode = RecordingMode.PAUSED
                stats.end_time = datetime.now()
                message = f"Auto-stopped at {AUTO_STOP_SECONDS}s"
                message_time = datetime.now()
            
            # Quality warning
            if QUALITY_CHECK_ENABLED and hands_detected < 2:
                cv2.putText(frame, "⚠ Both hands not visible!", (w//2 - 150, h - 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        # Clear temporary message after 3 seconds
        if message_time and (datetime.now() - message_time).total_seconds() > 3:
            message = ""
            message_time = None
        
        # Draw info panel
        countdown_val = 0
        if mode == RecordingMode.COUNTDOWN:
            countdown_val = COUNTDOWN_SECONDS - int((datetime.now() - countdown_start).total_seconds())
        draw_info_panel(frame, mode, stats, countdown_val, message)
        
        cv2.imshow('Digital Takumi - Enhanced Recorder', frame)
        
        # Store for velocity calculation
        if mode == RecordingMode.RECORDING:
            previous_frame_data = current_frame_data
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('r'):
            if mode == RecordingMode.IDLE or mode == RecordingMode.PAUSED or mode == RecordingMode.REVIEWING:
                if mode != RecordingMode.PAUSED:
                    recorded_frames = []
                    stats.reset()
                mode = RecordingMode.COUNTDOWN
                countdown_start = datetime.now()
                message = f"Get ready! Starting in {COUNTDOWN_SECONDS}..."
                message_time = datetime.now()
                print(f"\n>>> COUNTDOWN STARTED - Recording will begin in {COUNTDOWN_SECONDS} seconds!")
        
        elif key == ord('p'):
            if mode == RecordingMode.RECORDING:
                mode = RecordingMode.PAUSED
                stats.end_time = datetime.now()
                message = "Recording paused"
                message_time = datetime.now()
                print(f">>> RECORDING PAUSED - {stats.total_frames} frames captured")
            elif mode == RecordingMode.PAUSED:
                mode = RecordingMode.RECORDING
                stats.end_time = None
                message = "Recording resumed"
                message_time = datetime.now()
                print(">>> RECORDING RESUMED")
        
        elif key == ord(' '):  # Space bar
            if len(recorded_frames) > 0 and mode != RecordingMode.RECORDING:
                mode = RecordingMode.REVIEWING
                playback_recording(recorded_frames, landmarker)
                mode = RecordingMode.PAUSED
        
        elif key == ord('s'):
            if len(recorded_frames) >= MIN_RECORDING_FRAMES and mode != RecordingMode.RECORDING:
                if stats.end_time is None:
                    stats.end_time = datetime.now()
                
                # --- ASYNC SAVE START ---
                print(f"\n[Background] Saving {len(recorded_frames)} frames...")
                
                # 1. Thread the Save Operation
                # The thread handles all saving AND printing. We just fire and forget.
                save_thread = threading.Thread(
                    target=save_recording_with_metadata, 
                    args=(recorded_frames, stats, "golden_template")
                )
                save_thread.start()
                
                frames_file, metadata_file = save_recording_with_metadata(
                    recorded_frames, stats, "golden_template"
                )
                
                print(f"\n{'='*70}")
                print(f"✅ GOLDEN TEMPLATE SAVED SUCCESSFULLY!")
                print(f"{'='*70}")
                print(f"📁 Frames file: {frames_file}")
                print(f"📄 Metadata: {metadata_file}")
                print(f"📊 Statistics:")
                print(f"   - Total frames: {stats.total_frames}")
                print(f"   - Duration: {stats.get_duration():.2f} seconds")
                print(f"   - Quality score: {stats.get_quality_score():.1f}%")
                print(f"   - Frames with both hands: {stats.frames_with_both_hands}")
                print(f"   - Average confidence: {stats.avg_confidence:.2f}")
                print(f"{'='*70}\n")
                
                message = "Recording saved!"
                message_time = datetime.now()
                recorded_frames = []
                stats.reset()
                mode = RecordingMode.IDLE
            elif len(recorded_frames) < MIN_RECORDING_FRAMES:
                message = f"Need at least {MIN_RECORDING_FRAMES} frames!"
                message_time = datetime.now()
                print(f"⚠️  Recording too short! Need at least {MIN_RECORDING_FRAMES} frames.")
        
        elif key == ord('d'):
            if len(recorded_frames) > 0 and mode != RecordingMode.RECORDING:
                recorded_frames = []
                stats.reset()
                mode = RecordingMode.IDLE
                message = "Recording discarded"
                message_time = datetime.now()
                print(">>> RECORDING DISCARDED")
        
        elif key == ord('v'):
            show_velocity = not show_velocity
            message = f"Velocity vectors: {'ON' if show_velocity else 'OFF'}"
            message_time = datetime.now()
        
        elif key == ord('k'):
            show_skeleton = not show_skeleton
            message = f"Skeleton: {'ON' if show_skeleton else 'OFF'}"
            message_time = datetime.now()
        
        elif key == ord('l'):
            show_labels = not show_labels
            message = f"Joint labels: {'ON' if show_labels else 'OFF'}"
            message_time = datetime.now()

    cap.release()
    cv2.destroyAllWindows()

print("\n" + "="*70)
print("Session ended. Your enhanced golden templates are ready!")
print("="*70)