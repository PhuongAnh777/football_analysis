# ⚽ Football Tactical Analysis System (FTAS)

[cite_start]A Computer Vision-based system designed to analyze football match tactics from video footage, specifically optimized for V-League and student tournament data[cite: 2, 20].

---

## 🚀 Key Features

* [cite_start]**Object Detection & Tracking**: Real-time player, ball, and referee detection using **YOLOv8** [cite: 3] [cite_start]and **ByteTrack** to maintain unique IDs[cite: 5].
* [cite_start]**Team Classification**: Automatic jersey color recognition using **K-Means clustering**[cite: 6].
* **Tactical Metrics**:
    * [cite_start]**Compact Score**: Measures squad cohesion and connectivity using sliding time windows[cite: 8].
    * [cite_start]**Pressing Intensity**: Calculates player velocity and pressure radius around the ball[cite: 9].
    * [cite_start]**Formation Adherence**: Automatically identifies tactical setups (e.g., 4-4-2, 4-3-3) and detects strategic shifts[cite: 10].
* [cite_start]**Automated Insights**: Generates natural language tactical summaries based on calculated performance thresholds[cite: 11].
* [cite_start]**Web Dashboard**: Integrated **FastAPI** backend to visualize heatmaps, passing networks, and average position diagrams[cite: 13, 14].

---

## 🛠 Tech Stack

| Category | Technology |
| :--- | :--- |
| **AI/CV** | [cite_start]Python, PyTorch, OpenCV, YOLOv8, ByteTrack [cite: 2, 3, 5] |
| **Backend** | [cite_start]FastAPI [cite: 14] |
| **Frontend** | [cite_start]HTML, CSS, JavaScript [cite: 14] |
| **Visualization** | [cite_start]Matplotlib, Plotly  |

---

## 📅 Development Roadmap

### Phase 1: Basic Pipeline (Weeks 1-3)
* [cite_start]Environment setup and data collection (V-League/Student leagues)[cite: 2].
* [cite_start]Deployment of YOLOv8 and ByteTrack integration[cite: 3, 5].

### Phase 2: Tactical Algorithms (Weeks 4-7)
* [cite_start]Development of **Compact Score** and **Pressing Intensity** modules[cite: 8, 9].
* [cite_start]Implementation of **Formation Recognition** and Automated Insights[cite: 10, 11].

### Phase 3: Visualization & Web System (Weeks 8-10)
* [cite_start]Creating tactical maps (Heatmaps, Passing networks).
* [cite_start]Full system integration: Video upload -> CV processing -> Web results[cite: 15].

### Phase 4: Evaluation & Reporting (Weeks 11-13)
* [cite_start]Testing with 3–5 real-world videos and measuring **mAP, Precision, and Recall**[cite: 17, 18].
* [cite_start]Analyzing deviations caused by Vietnamese broadcasting standards[cite: 20].

---

## 📊 Evaluation Metrics
[cite_start]The system is evaluated based on its ability to handle non-standard camera angles in Vietnamese stadiums [cite: 20][cite_start], aiming for high precision in player tracking and formation detection[cite: 18].

[cite_start]**Deadline**: June 28, 2026[cite: 21].
