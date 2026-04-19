# ⚽ Football Tactical Analysis System (FTAS)

A Computer Vision-based system designed to analyze football match tactics from video footage, specifically optimized for V-League and student tournament data.

---

## 🚀 Key Features

* **Object Detection & Tracking**: Real-time player, ball, and referee detection using **YOLOv8** and **ByteTrack** to maintain unique IDs.
* **Team Classification**: Automatic jersey color recognition using **K-Means clustering**.
* **Tactical Metrics**:
    * **Compact Score**: Measures squad cohesion and connectivity using sliding time windows.
    * **Pressing Intensity**: Calculates player velocity and pressure radius around the ball.
    * **Formation Adherence**: Automatically identifies tactical setups (e.g., 4-4-2, 4-3-3) and detects strategic shifts.
* **Automated Insights**: Generates natural language tactical summaries based on calculated performance thresholds.
* **Web Dashboard**: Integrated **FastAPI** backend to visualize heatmaps, passing networks, and average position diagrams.

---

## 🛠 Tech Stack

| Category | Technology |
| :--- | :--- |
| **AI/CV** | Python, PyTorch, OpenCV, YOLOv8, ByteTrack |
| **Backend** | FastAPI |
| **Frontend** | HTML, CSS, JavaScript |
| **Visualization** | Matplotlib, Plotly |

---

## 📅 Development Roadmap

### Phase 1: Basic Pipeline (Weeks 1-3)
* Environment setup and data collection (V-League/Student leagues).
* Deployment of YOLOv8 and ByteTrack integration.

### Phase 2: Tactical Algorithms (Weeks 4-7)
* Development of **Compact Score** and **Pressing Intensity** modules.
* Implementation of **Formation Recognition** and Automated Insights.

### Phase 3: Visualization & Web System (Weeks 8-10)
* Creating tactical maps (Heatmaps, Passing networks).
* Full system integration: Video upload -> CV processing -> Web results.

### Phase 4: Evaluation
