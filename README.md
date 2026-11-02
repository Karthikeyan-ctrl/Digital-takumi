# Digital Takumi: AI-Powered Heritage & Tacit Skill Transfer System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-orange.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](#license)

**Digital Takumi** is an augmented reality and computer-vision framework designed to capture, quantify, and transfer tacit motor skills and artisan techniques. Inspired by Japanese master craftspeople (*Takumi*), this system digitizes fine hand biomechanics, spatial trajectories, and timing to bridge the generational succession gap in precision manufacturing and traditional crafts.

---

## Table of Contents

- [The Problem & Vision](#the-problem--vision)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
  - [1. 3D Hand Tracking & 1€ Signal Stabilization](#1-3d-hand-tracking--1-signal-stabilization)
  - [2. Biomechanical & Kinematic Digitization](#2-biomechanical--kinematic-digitization)
  - [3. Golden Template Recording (`main.py`)](#3-golden-template-recording-mainpy)
  - [4. Real-Time AR Ghost Alignment & Guidance (`digital_takumi_feedback.py`)](#4-real-time-ar-ghost-alignment--guidance-digital_takumi_feedbackpy)
  - [5. Dynamic Time Warping (DTW) & Biometric Scoring](#5-dynamic-time-warping-dtw--biometric-scoring)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
  - [Recording an Expert Session](#recording-an-expert-session)
  - [Apprentice Practice & Feedback](#apprentice-practice--feedback)
- [Mathematical & Kinematic Formulations](#mathematical--kinematic-formulations)
- [Roadmap](#roadmap)

---

## The Problem & Vision

Traditional craftsmanship (pottery, woodworking, metal spinning, precision die-making, surgical motions) relies heavily on **tacit knowledge**—subtle muscle memory, wrist tilt angles, finger flexion, and movement cadence acquired over decades of direct observation.

As veteran artisans retire (often referred to as the *"2025 Cliff"* demographic crisis in Japanese manufacturing), this unwritten expertise risks being permanently lost. **Digital Takumi** converts subjective "feel" into objective 3D kinematic data:
- **Record**: Masters demonstrate techniques in front of a standard camera.
- **Digitize**: The system computes 3D joint angles, wrist Euler rotations (Roll/Pitch/Yaw), finger spreads, and velocity vectors.
- **Train**: Apprentices practice with real-time AR "ghost" overlays matching their posture, backed by speed-invariant Dynamic Time Warping (DTW) and audio-visual biometric feedback.

---

## System Architecture

```text
               +------------------------------------+
               |      Monocular Camera Stream       |
               +------------------------------------+
                                 |
                                 v
               +------------------------------------+
               |  MediaPipe HandLandmarker (3D lm)  |
               +------------------------------------+
                                 |
                                 v
               +------------------------------------+
               |   1€ Adaptive Low-Pass Filtering   |
               | (Zero-lag jitter cancellation)     |
               +------------------------------------+
                                 |
                                 v
               +------------------------------------+
               |      Skeletal Digitizer Engine     |
               | - Joint Flexion (MCP/PIP/DIP)      |
               | - Finger Spread / Abduction        |
               | - 3D Wrist Euler Orientation       |
               +------------------------------------+
                     /                        \
                    /                          \
                   v                            v
  [Mode 1: main.py]             [Mode 2: digital_takumi_feedback.py]
  +--------------------------+   +------------------------------------+
  | Golden Template Recorder |   | AR Ghost Overlay & Alignment       |
  | - Quality validation     |   | - 16:9 metric-corrected rotation   |
  | - Live Biometric HUD     |   | - Dynamic Time Warping (DTW)       |
  | - Velocity vector engine |   | - Acceleration & jerk analysis     |
  | - Serialized .npy + JSON |   | - Multi-sensory feedback & scoring |
  +--------------------------+   +------------------------------------+
```

---

## How It Works

### 1. 3D Hand Tracking & 1€ Signal Stabilization
- Uses **MediaPipe Tasks Vision** (`HandLandmarker`) with `hand_landmarker.task` to extract 21 three-dimensional anatomical landmarks per hand.
- Raw vision landmarks can jitter when stationary and exhibit latency when moving quickly. To solve this, a **1€ (One Euro) Filter** is applied to each coordinate:
  - When the hand is still: High smoothing eliminates micro-tremors and tracking noise.
  - When the hand moves rapidly: Dynamic cutoff frequency scales up automatically, providing sub-millisecond responsiveness with near-zero lag.

### 2. Biomechanical & Kinematic Digitization
The `SkeletalDigitizer` class extracts physical biomechanics invariant to hand size and camera distance:
- **Spatial Normalization**: Centers coordinates at the wrist joint (ID 0) and scales uniformly against the palm reference length (Wrist to Middle MCP joint).
- **Joint Flexion (Knuckle bends)**: Computes interior angles across MCP, PIP, and DIP/IP joints for all 5 digits.
- **Finger Abduction (Spread)**: Computes angular separation between adjacent metacarpal vectors (`Index-Mid`, `Mid-Ring`, `Ring-Pinky`).
- **Wrist Euler Orientation**:
  - Derives a local orthonormal basis from the palm plane ($v_{\text{wrist} \to \text{index}}$ and $v_{\text{wrist} \to \text{pinky}}$).
  - Calculates global **Roll** (tilt), **Pitch** (up/down), and **Yaw** (lateral rotation) relative to the camera frame.

### 3. Golden Template Recording (`main.py`)
- Guides the master through an intuitive recording state machine (`IDLE` $\to$ `COUNTDOWN` $\to$ `RECORDING` $\to$ `PAUSED` $\to$ `REVIEWING`).
- Displays an interactive **Biometric HUD** showing real-time joint flexion meters and color-coded stress levels.
- Draws real-time velocity vector arrows indicating movement trajectory and speed.
- Automatically computes session statistics (both-hands visibility percentage, average detection confidence, duration).
- Saves synchronized `.npy` kinematic arrays alongside a detailed JSON metadata log.

### 4. Real-Time AR Ghost Alignment & Guidance (`digital_takumi_feedback.py`)
- Loads the master's golden template and projects a translucent golden ghost skeleton over the apprentice's live hand.
- **Aspect-Ratio Metric Correction**: Standard camera frames (16:9) distort rotations when computed in normalized camera coordinates. The engine maps landmarks to physical aspect-ratio space before applying 3D rotation matrices, preventing hand-stretching artifacts.
- **Visual Feedback**:
  - Gold skeleton: Master reference.
  - Green skeleton: Apprentice live hand.
  - Red circles: Joints exceeding acceptable spatial deviation thresholds.

### 5. Dynamic Time Warping (DTW) & Biometric Scoring
Apprentices rarely perform motions at the exact millisecond pace of an expert. Linear frame-by-frame comparison would penalize natural speed variances.
- **Dynamic Time Warping (DTW)** computes the optimal non-linear temporal alignment between apprentice and master sequences.
- **Form Evaluation**: Compares joint flexion and wrist angles strictly across aligned time steps.
- **Kinematic Jerk / Snap Detection**: Computes second-order derivatives (acceleration) to flag involuntary twitches or sudden erratic movements.
- **Grading Output**: Yields a 0–100% mastery score, average angular error, list of top failing joints, and wrist stability metrics.
- **Audio Cues**: Thread-safe acoustic signals alert the user to synchronization snaps, successful milestones, or excessive deviations.

---

## Project Structure

```text
Digital-Takumi/
├── digital_takumi_feedback.py          # Real-time AR ghost overlay, DTW comparison & scoring
├── main.py                             # Master recorder, 1€ filter, biometric HUD & template exporter
├── hand_landmarker.task                # MediaPipe pre-trained 3D hand landmark model
├── golden_template_20260104_090115.npy # Recorded master session data (frames & landmarks)
├── golden_template_20260104_090115_metadata.json # Metadata (FPS, duration, confidence, resolution)
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git ignore rules for environments and temporary files
└── README.md                           # Documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- A functional webcam (720p @ 30 FPS or higher recommended)
- Windows, macOS, or Linux

### 1. Clone the Repository
```bash
git clone https://github.com/Karthikeyan-ctrl/Digital-takumi.git
cd Digital-takumi
```

### 2. Create and Activate a Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify MediaPipe Model Asset
Ensure `hand_landmarker.task` is in the root directory. If missing, it can be downloaded directly from Google MediaPipe:
```bash
# Optional download if not already present
curl -o hand_landmarker.task -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

---

## Usage Guide

### Recording an Expert Session
Run the recording interface to create a master template:
```bash
python main.py
```

#### Controls & Shortcuts:
| Key | Action |
| :--- | :--- |
| `R` | Start countdown and begin recording |
| `P` | Pause / Resume active recording |
| `Space` | Finish recording and enter Review mode |
| `S` | Save recorded session to `.npy` and metadata `.json` |
| `D` | Discard current recording and reset |
| `V` | Toggle velocity arrows |
| `K` | Toggle skeleton overlay |
| `L` | Toggle joint landmark index numbers |
| `Q` | Exit application |

Saved templates are named in the format:
`golden_template_<YYYYMMDD_HHMMSS>.npy`

---

### Apprentice Practice & Feedback
To compare a live performance against a saved golden template:

1. Open `digital_takumi_feedback.py` and set `GOLDEN_TEMPLATE_FILE` to your desired template file (or keep the default included template):
   ```python
   GOLDEN_TEMPLATE_FILE = "golden_template_20260104_090115.npy"
   ```
2. Start the feedback application:
   ```bash
   python digital_takumi_feedback.py
   ```
3. Perform the artisan motion aligned with the translucent golden ghost.
4. Press `Space` to finish the session and review the comprehensive terminal breakdown:
   - **Form Score (0–100%)**
   - **Joints with Highest Deviation (> 15°)**
   - **Wrist Roll / Pitch / Yaw Drift**
   - **Temporal Smoothness & Acceleration Profile**

---

## Mathematical & Kinematic Formulations

### Joint Flexion Angle
For consecutive bones represented by 3D vectors $\mathbf{u}$ and $\mathbf{v}$:

$$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}, \quad \theta = \arccos\left(\text{clamp}(\cos(\theta), -1.0, 1.0)\right)$$

### 1€ Filter (One Euro Filter)
Smoothing factor $\alpha$ determined by sampling interval $T_e$ and dynamic cutoff frequency $f_c$:

$$\hat{x}_i = \alpha x_i + (1 - \alpha) \hat{x}_{i-1}, \quad \alpha = \frac{2\pi f_c T_e}{2\pi f_c T_e + 1}$$

$$f_c = f_{c,\text{min}} + \beta |\hat{\dot{x}}_i|$$

### Dynamic Time Warping (DTW)
Given master sequence $G = (g_1, \dots, g_n)$ and apprentice sequence $L = (l_1, \dots, l_m)$, the cumulative distance matrix $D$ is defined recursively as:

$$D(i, j) = d(g_i, l_j) + \min \begin{cases} D(i-1, j) & \text{(apprentice hesitation)} \\ D(i, j-1) & \text{(expert hesitation)} \\ D(i-1, j-1) & \text{(synchronized step)} \end{cases}$$

---

## Roadmap

- [ ] Support for Full-Body Posture Tracking via MediaPipe Pose (capturing torso alignment, elbow height, and shoulder balance).
- [ ] Direct export to 3D formats (FBX, BVH) for Blender and robotics teleoperation.
- [ ] Wearable haptic motor integration (ESP32 Bluetooth vibrotactile feedback on wrists/fingers).
- [ ] Web-based visualization dashboard using Three.js / WebGL.

---

## License

This project is licensed under the [MIT License](LICENSE).
