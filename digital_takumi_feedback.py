import cv2
import mediapipe as mp
import numpy as np
import os
import math
import threading
import platform
import time

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

# Joint names (for feedback)
JOINT_NAMES = [
    "Wrist",
    "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb Tip",
    "Index MCP", "Index PIP", "Index DIP", "Index Tip",
    "Middle MCP", "Middle PIP", "Middle DIP", "Middle Tip",
    "Ring MCP", "Ring PIP", "Ring DIP", "Ring Tip",
    "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky Tip"
]

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
    
    def align_hand_to_reference(self, target_hand, reference_hand):
        """
        Rotates 'target_hand' (Expert) to match the orientation of 'reference_hand' (User).
        Fixes the 'Stretching Bug' by correcting for Aspect Ratio (16:9).
        """
        # 1. Setup Aspect Ratio Correction (Assume 1280x720)
        AR = 1280.0 / 720.0
        
        def to_metric(lm):
            # Scale X to match Y's physical scale
            return np.array([lm[0] * AR, lm[1], lm[2]])
            
        def from_metric(pt):
            # Scale X back to normalized coordinates
            return np.array([pt[0] / AR, pt[1], pt[2]])

        # 2. Convert both hands to "Real World" (Metric) space
        # This prevents the rotation from stretching the fingers
        target_metric = np.array([to_metric(p) for p in target_hand])
        ref_metric = np.array([to_metric(p) for p in reference_hand])

        # 3. Get Basis Vectors (Orientation) in Metric Space
        def get_basis(lm):
            wrist = lm[0]
            index_mcp = lm[5]
            pinky_mcp = lm[17]
            
            v1 = index_mcp - wrist
            v1 /= np.linalg.norm(v1)
            
            v2 = pinky_mcp - wrist
            v2 /= np.linalg.norm(v2)
            
            palm_normal = np.cross(v1, v2)
            palm_normal /= np.linalg.norm(palm_normal)
            
            x_axis = np.cross(palm_normal, v1)
            x_axis /= np.linalg.norm(x_axis)
            
            z_axis = np.cross(x_axis, palm_normal)
            
            return np.column_stack((x_axis, palm_normal, z_axis)), wrist

        try:
            R_target, origin_target = get_basis(target_metric)
            R_ref, origin_ref = get_basis(ref_metric)
        except ValueError:
            return target_hand 
        
        # 4. Calculate Rotation Matrix
        R_diff = R_ref @ R_target.T
        
        # 5. Apply Rotation & Position Shift
        aligned_landmarks = []
        for pt in target_metric:
            # Shift to 0,0,0
            pt_centered = pt - origin_target
            # Rotate
            pt_rotated = R_diff @ pt_centered
            # Shift to User's Wrist
            pt_final_metric = pt_rotated + origin_ref
            
            # Convert back to Camera Space (Normalized)
            pt_final = from_metric(pt_final_metric)
            aligned_landmarks.append(pt_final)
            
        return aligned_landmarks

# ====================== CONFIGURATION ======================
GOLDEN_TEMPLATE_FILE = "golden_template_20260104_090115.npy"  # <-- CHANGE THIS
PLAYBACK_SPEED = 1.0  # Frames to advance per loop (1.0 = original speed)
EXPERT_COLOR = (255, 215, 0)   # Gold
LIVE_COLOR = (0, 255, 0)      # Green
EXPERT_ALPHA = 0.5            # Expert overlay transparency
ERROR_THRESHOLD_PX = 20       # For real-time red circles
SNAP_THRESHOLD = 0.01         # Threshold for detecting snap movements
# ===========================================================

# Load golden template
if not os.path.exists(GOLDEN_TEMPLATE_FILE):
    raise FileNotFoundError(f"Golden template not found: {GOLDEN_TEMPLATE_FILE}")
golden_data = np.load(GOLDEN_TEMPLATE_FILE, allow_pickle=True)
print(f"Loaded golden template: {len(golden_data)} frames")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

def draw_hand(frame, landmarks, color, thickness=2, alpha=1.0):
    h, w, _ = frame.shape
    pts = []
    for lm in landmarks:
        x_px = int(lm[0] * w)
        y_px = int(lm[1] * h)
        pts.append((x_px, y_px))
    
    overlay = frame.copy()
    for a, b in HAND_CONNECTIONS:
        cv2.line(overlay, pts[a], pts[b], color, thickness)
    for pt in pts:
        cv2.circle(overlay, pt, 5, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

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

def normalize_hand(landmarks):
    """
    Centers the hand at the wrist and scales it to a unit size.
    Input: np.array of shape (21, 3)
    """
    # 1. Translation: Center wrist at 0,0,0
    wrist = landmarks[0]
    centered = landmarks - wrist

    # 2. Scale: Normalize based on size (Wrist to Middle Finger MCP)
    # Joint 0 is Wrist, Joint 9 is Middle MCP
    scale_ref = np.linalg.norm(centered[9]) 
    if scale_ref == 0: return centered # Prevent divide by zero
    
    normalized = centered / scale_ref
    return normalized

# Simple cross-platform beep
def play_sound(sound_type):
    """
    Types: 'success', 'error', 'snap', 'start'
    """
    def _worker():
        system = platform.system()
        try:
            if system == "Windows":
                import winsound
                if sound_type == 'success':
                    winsound.Beep(1000, 150) # High pitch, short
                    winsound.Beep(1500, 150)
                elif sound_type == 'error':
                    winsound.Beep(300, 300)  # Low pitch, long
                elif sound_type == 'snap':
                    winsound.Beep(2000, 50)  # Very short click
                elif sound_type == 'start':
                    winsound.Beep(600, 200)
            else:
                # Mac/Linux fallback (print bell)
                print('\a') 
        except:
            pass
            
    # Run in thread so it doesn't freeze the video
    threading.Thread(target=_worker).start()

def frame_dist(frame1, frame2):
    """Distance between two frames: avg Euclidean per joint, matching hands by handedness"""
    hands1 = {h['handedness']: np.array(h['landmarks']) for h in frame1['hands']}
    hands2 = {h['handedness']: np.array(h['landmarks']) for h in frame2['hands']}
    
    total_dist = 0.0
    count = 0
    
    for side in ['Left', 'Right']:
        if side in hands1 and side in hands2:
            lm1 = hands1[side]
            lm2 = hands2[side]
            norm1 = normalize_hand(lm1)
            norm2 = normalize_hand(lm2)
            dists = np.linalg.norm(norm1 - norm2, axis=1)
            total_dist += np.sum(dists)
            count += len(dists)
    
    if count == 0:
        return np.inf
    return total_dist / count

def dtw(s1, s2, dist_func):
    """DTW with path"""
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dist_func(s1[i-1], s2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    
    # Backtrack path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        min_val = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
        if dtw_matrix[i-1, j-1] == min_val:
            i -= 1
            j -= 1
        elif dtw_matrix[i-1, j] == min_val:
            i -= 1
        else:
            j -= 1
    path.reverse()
    
    return dtw_matrix[n, m], path

def compute_velocities(sequence):
    """Compute velocity for each joint across frames"""
    velocities = []
    for i in range(len(sequence) - 1):
        frame_vel = {'hands': []}
        curr_frame = sequence[i]
        next_frame = sequence[i + 1]
        
        # Match hands by handedness
        hands_curr = {h['handedness']: np.array(h['landmarks']) for h in curr_frame['hands']}
        hands_next = {h['handedness']: np.array(h['landmarks']) for h in next_frame['hands']}
        
        for side in ['Left', 'Right']:
            if side in hands_curr and side in hands_next:
                # Velocity = displacement between frames (normalized coords per frame)
                vel = hands_next[side] - hands_curr[side]
                frame_vel['hands'].append({
                    'handedness': side,
                    'velocities': vel  # 21x3 array
                })
        
        velocities.append(frame_vel)
    
    return velocities

def compute_accelerations(velocities):
    """Compute acceleration from velocity sequences"""
    accelerations = []
    for i in range(len(velocities) - 1):
        frame_acc = {'hands': []}
        curr_vel = velocities[i]
        next_vel = velocities[i + 1]
        
        hands_curr = {h['handedness']: np.array(h['velocities']) for h in curr_vel['hands']}
        hands_next = {h['handedness']: np.array(h['velocities']) for h in next_vel['hands']}
        
        for side in ['Left', 'Right']:
            if side in hands_curr and side in hands_next:
                # Acceleration = change in velocity
                acc = hands_next[side] - hands_curr[side]
                frame_acc['hands'].append({
                    'handedness': side,
                    'accelerations': acc  # 21x3 array
                })
        
        accelerations.append(frame_acc)
    
    return accelerations

def detect_snap_movements(accelerations, threshold=0.01):
    """Identify sudden movements (high acceleration spikes)"""
    snap_frames = []
    for frame_idx, frame_acc in enumerate(accelerations):
        for hand_data in frame_acc['hands']:
            acc_magnitudes = np.linalg.norm(hand_data['accelerations'], axis=1)
            max_acc = np.max(acc_magnitudes)
            if max_acc > threshold:
                joint_idx = np.argmax(acc_magnitudes)
                snap_frames.append({
                    'frame': frame_idx,
                    'hand': hand_data['handedness'],
                    'joint': joint_idx,
                    'magnitude': max_acc
                })
    return snap_frames

def compute_feedback(golden_seq, live_seq, digitizer):
    """
    Advanced Feedback: Compares PHYSICS (Angles) + TIMING (Velocity).
    """
    if len(live_seq) < 5 or len(golden_seq) < 5:
        return "Sequence too short for analysis."
    
    # 1. DTW Alignment (Keep using Euclidean distance for alignment, it's robust)
    # We align based on normalized geometry to match the timeline
    total_dist, path = dtw(golden_seq, live_seq, frame_dist)
    
    # 2. Physics Analysis Containers
    angle_diffs = {} # Stores error per joint angle (e.g., "Index_MCP_Flexion": [5, 2, 6...])
    wrist_diffs = {"roll": [], "pitch": [], "yaw": []}
    
    # 3. Traverse the Aligned Path
    for g_idx, l_idx in path:
        g_frame = golden_seq[g_idx]
        l_frame = live_seq[l_idx]
        
        # Get hands
        hands_g = {h['handedness']: h for h in g_frame['hands']}
        hands_l = {h['handedness']: h for h in l_frame['hands']}
        
        for side in ['Left', 'Right']:
            if side in hands_g and side in hands_l:
                # --- A. ANGLE COMPARISON (The "True" Score) ---
                # Check if Golden Template actually has angles (backward compatibility)
                if 'joint_angles' in hands_g[side]:
                    ang_g = hands_g[side]['joint_angles']
                else:
                    # If loading an old recording, compute on the fly
                    ang_g = digitizer.compute_hand_angles(hands_g[side]['landmarks'])

                # Live frame angles (we compute these now if missing)
                if 'joint_angles' in hands_l[side]:
                    ang_l = hands_l[side]['joint_angles']
                else:
                    ang_l = digitizer.compute_hand_angles(hands_l[side]['landmarks'])
                
                # Compare every angle key
                for key in ang_g:
                    if "wrist" in key: continue # Handle wrist separately
                    
                    diff = abs(ang_g[key] - ang_l[key])
                    
                    if key not in angle_diffs: angle_diffs[key] = []
                    angle_diffs[key].append(diff)

                # --- B. WRIST ORIENTATION COMPARISON ---
                # Compare Roll/Pitch/Yaw if available
                if 'wrist_roll' in ang_g and 'wrist_roll' in ang_l:
                    wrist_diffs['roll'].append(abs(ang_g['wrist_roll'] - ang_l['wrist_roll']))
                    wrist_diffs['pitch'].append(abs(ang_g['wrist_pitch'] - ang_l['wrist_pitch']))
                    wrist_diffs['yaw'].append(abs(ang_g['wrist_yaw'] - ang_l['wrist_yaw']))

    # 4. Generate Report
    feedback_str = f"\n{'='*60}\n"
    feedback_str += f"{'ADVANCED BIOMETRIC FEEDBACK':^60}\n"
    feedback_str += f"{'='*60}\n\n"
    
    # SCORING: 
    # Perfect = 0 degrees error. 
    # Acceptable = < 15 degrees error.
    
    # A. Finger Mechanics
    total_angle_error = 0
    count = 0
    worst_joints = []
    
    for key, diffs in angle_diffs.items():
        avg_err = np.mean(diffs)
        total_angle_error += avg_err
        count += 1
        if avg_err > 15.0: # Threshold for "Bad Form"
            worst_joints.append((avg_err, key))
            
    avg_bio_score = total_angle_error / max(count, 1)
    
    # Calculate Grade (0-100%)
    # If average error is 0 deg -> 100%. If 30 deg -> 0%.
    final_score = max(0, 100 - (avg_bio_score * 3.3))
    
    feedback_str += f"🏆 MUSCLE MEMORY SCORE: {final_score:.1f}%\n"
    feedback_str += f"   (Avg Joint Error: {avg_bio_score:.1f}°)\n\n"
    
    if final_score > 85:
        feedback_str += "   ✅ EXCELLENT! Your hand shape matches the expert.\n"
    elif final_score > 60:
        feedback_str += "   ⚠️ GOOD, but watch your finger curvature.\n"
    else:
        feedback_str += "   ❌ NEEDS PRACTICE. Your hand shape is incorrect.\n"

    # B. Specific Corrections
    if worst_joints:
        feedback_str += f"\n🔧 FIX THESE JOINTS:\n"
        # Sort by worst error
        worst_joints.sort(key=lambda x: x[0], reverse=True)
        for err, name in worst_joints[:3]:
            feedback_str += f"   • {name}: Off by {err:.1f}°\n"
            if "Flexion" in name:
                feedback_str += "     (Try bending/straightening this finger)\n"
            elif "Spread" in name:
                feedback_str += "     (Check gap between fingers)\n"

    # C. Wrist Alignment
    r_err = np.mean(wrist_diffs['roll']) if wrist_diffs['roll'] else 0
    p_err = np.mean(wrist_diffs['pitch']) if wrist_diffs['pitch'] else 0
    y_err = np.mean(wrist_diffs['yaw']) if wrist_diffs['yaw'] else 0
    
    feedback_str += f"\n📐 WRIST ALIGNMENT:\n"
    feedback_str += f"   • Roll (Tilt):  {'✅ OK' if r_err < 15 else f'❌ Fix Tilt ({r_err:.0f}° err)'}\n"
    feedback_str += f"   • Pitch (Up/Dn): {'✅ OK' if p_err < 15 else f'❌ Fix Angle ({p_err:.0f}° err)'}\n"
    feedback_str += f"   • Yaw (Turn):   {'✅ OK' if y_err < 15 else f'❌ Fix Rotation ({y_err:.0f}° err)'}\n"

    feedback_str += f"\n{'='*60}\n"
    return feedback_str

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    # Match recorder settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    session_active = False
    frame_idx = 0
    live_frames = []
    total_frames = len(golden_data)
    
    print("\n=== Digital Takumi - Enhanced DTW Feedback Mode ===")
    print("Camera: 1280x720 @ 30fps (matching recorder)")
    print("Live: Green | Expert: Gold | Red circles = real-time errors")
    print("Yellow arrows = velocity vectors (wrist only)")
    print("Press 's' to start session (mimic the expert)")
    print("Press 'r' to reset | 'q' to quit\n")

    digitizer = SkeletalDigitizer()
    is_aligned = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror for intuitive movement
        frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        h, w, _ = frame.shape
        
        current_live_data = {'hands': []}

        # 1. Process Live Hands
        if results.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
                landmarks = hand_landmarks.landmark if hasattr(hand_landmarks, 'landmark') else hand_landmarks
                lm_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
                
                # --- MIRROR LABEL FIX ---
                raw_label = results.handedness[hand_idx][0].category_name if results.handedness else "?"
                if raw_label == "Left":
                    handedness = "Right"
                elif raw_label == "Right":
                    handedness = "Left"
                else:
                    handedness = raw_label
                # ------------------------

                current_live_data['hands'].append({
                    'handedness': handedness,
                    'landmarks': lm_list
                })
                
                # Draw Live HUD for the first detected hand
                if hand_idx == 0:
                    live_angles = digitizer.compute_hand_angles(lm_list)
                    wrist_ori = digitizer.compute_wrist_orientation(lm_list)
                    live_angles.update(wrist_ori)
                    draw_biometric_hud(frame, live_angles, handedness)
                    # ---  REAL-TIME AUDIO FEEDBACK ---
                    # 1. Get the current Expert Angles to compare against
                    if frame_idx < total_frames:
                        expert_frame = golden_data[frame_idx]
                        # Find the expert hand that matches our current live hand (Left/Right)
                        expert_hand_data = next((h for h in expert_frame['hands'] if h['handedness'] == handedness), None)
                        
                        if expert_hand_data:
                            # Extract or compute expert angles
                            if 'joint_angles' in expert_hand_data:
                                expert_angles = expert_hand_data['joint_angles']
                            else:
                                expert_angles = digitizer.compute_hand_angles(expert_hand_data['landmarks'])

                            # 2. Calculate "Error Score" (Average angle difference)
                            total_diff = 0
                            count = 0
                            for k in live_angles:
                                if k in expert_angles and "wrist" not in k: # Compare finger bends only
                                    total_diff += abs(live_angles[k] - expert_angles[k])
                                    count += 1
                            
                            if count > 0:
                                error_score = total_diff / count # Average error in degrees
                                
                                # 3. Trigger Audio (The "Clicker" Logic)
                                # Threshold: < 15 degrees means you matched the shape!
                                if error_score < 15 and not is_aligned:
                                    # play_sound('success')  <-- Uncomment after adding play_sound function
                                    print("PING! (Aligned)") # Visual debug
                                    is_aligned = True
                                    
                                elif error_score > 25: # Hysteresis: Reset only if you move far away
                                    is_aligned = False
                    # -------------------------------------

        # 2. Draw Expert Ghost (Magnetic AR)
        # We draw this regardless of 'session_active' so you can practice aligning before you start.
        if frame_idx < total_frames:
            expert_frame = golden_data[frame_idx]
            
            # Find live hands to magnetize to
            hands_live = {h['handedness']: h['landmarks'] for h in current_live_data['hands']}
            
            for expert_hand in expert_frame['hands']:
                exp_lm = expert_hand['landmarks']
                side = expert_hand['handedness']
                
                # MAGNETIC LOGIC: If user hand is visible, snap ghost to it
                if side in hands_live:
                    user_lm = hands_live[side]
                    aligned_ghost = digitizer.align_hand_to_reference(exp_lm, user_lm)
                    draw_hand(frame, aligned_ghost, EXPERT_COLOR, thickness=2, alpha=EXPERT_ALPHA)
                else:
                    # Fallback: If no user hand, draw static ghost
                    draw_hand(frame, exp_lm, EXPERT_COLOR, thickness=2, alpha=EXPERT_ALPHA)

        # 3. Session Logic (Only advance frames if active)
        if session_active:
            # Advance frame
            live_frames.append(current_live_data)
            frame_idx = int(frame_idx + PLAYBACK_SPEED)
            
            # Check for end of session
            if frame_idx >= total_frames:
                session_active = False
                feedback = compute_feedback(golden_data, live_frames, digitizer)
                print(feedback)
                print("\nSession complete. Press 's' for another try!")
                frame_idx = 0 # Reset for next preview

        # UI Text
        status = "ACTIVE - Mimic the expert!" if session_active else "Ready - Press 's' to start"
        status_color = (0, 0, 255) if session_active else (0, 255, 0)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        cv2.putText(frame, "S: Start | R: Reset | Q: Quit", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        cv2.imshow('Digital Takumi - Enhanced Feedback', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not session_active:
            session_active = True
            live_frames = []
            print(">>> SESSION STARTED - Follow the expert overlay!")
        elif key == ord('r'):
            session_active = False
            frame_idx = 0
            live_frames = []
            print("Reset complete.")

    cap.release()
    cv2.destroyAllWindows()

print("Program ended. Great practice!")