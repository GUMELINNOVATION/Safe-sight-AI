# 🦺 Enterprise Real-Time PPE Compliance & Risk Monitoring System
### Powered by YOLOv8 · Built with OpenCV · Eastern Mediterranean University

> An AI-driven safety monitoring system that autonomously detects workers and verifies PPE compliance (helmets & vests) in real time — replacing manual safety auditing with proactive, automated risk prevention.

---

## 🎯 Project Overview

Workplace accidents in construction and industrial environments result in severe injuries and massive financial losses. This system introduces an **enterprise-grade, real-time AI monitoring solution** that:

- 📷 Captures a **live camera feed** directly on a local machine (no cloud, no internet required)
- 🧠 Runs a **Dual-Model YOLOv8 inference pipeline** to detect workers and their safety equipment every frame
- ⚠️ Uses a **Mathematical Logic Layer** to deduce PPE violations without needing a "no-helmet" class
- 📊 Displays a live **Heads-Up Display (HUD)** with risk status, worker counts, FPS, and alerts
- 📝 **Auto-logs all violations** to a timestamped CSV file for management reporting

---

## 🧠 Dual-Model Architecture

The system uses **two YOLOv8 models running in parallel** on every frame:

| Model | Source | Responsibility |
|-------|--------|----------------|
| `yolov8n.pt` | Official Ultralytics (COCO) | Detects **persons** — highly reliable (87%+ confidence) |
| `best.pt` | Custom-trained | Detects **helmets** and **vests** |

> **Why two models?** The custom `best.pt` model is specialized for PPE equipment but proved unreliable for person detection in varied environments. `yolov8n.pt` solves this with industry-standard accuracy.

---

## 🔍 Detection Classes & Colors

| Model | Class | Box Color |
|-------|-------|-----------|
| `yolov8n.pt` | `person` | 🟠 Orange |
| `best.pt` | `helmet` | 🟢 Green |
| `best.pt` | `vest` | 🩵 Cyan-Teal |

---

## 🧮 Violation Logic Layer

Violations are deduced mathematically — no extra training needed:

```
missing_helmets  = max(0, person_count - helmet_count)
missing_vests    = max(0, person_count - vest_count)
total_violations = missing_helmets + missing_vests
```

This approach is **faster** and proves **deep engineering problem-solving** — no need to train separate "no-helmet" or "no-vest" classes.

---

## 🚦 Risk States

| State | Trigger | HUD Color | Effect |
|-------|---------|-----------|--------|
| **STANDBY** | No workers detected | ⚪ Grey | Monitoring active message |
| **SAFE** | All workers have full PPE | 🟢 Green | Full compliance message |
| **CRITICAL** | Any PPE missing | 🔴 Red | Flashing red border + violation count + CSV log |

---

## 📁 Project Structure

```
Yolo--main/
│
├── 📂 Web_Version_Archive/        ← Flask web dashboard (archived)
│   ├── app.py
│   └── templates/
│       ├── index.html
│       └── monitor.html
│
├── 🧠 best.pt                     ← Custom YOLOv8 model (helmet + vest)
├── 🎬 live_demo.py                ← Main entry point — OpenCV live demo
├── 📄 README.md
├── 📋 requirements.txt
└── 📊 safety_report.csv           ← Auto-generated violation log
```

> `yolov8n.pt` is downloaded automatically on first run from Ultralytics servers.

---

## ⚙️ Installation

**1. Clone the repository:**
```bash
git clone https://github.com/Aman-saeed-mohamed/Yolo-.git
cd Yolo-
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Live Demo

```bash
python live_demo.py
```

- On first run, `yolov8n.pt` (~6MB) downloads automatically.
- Press **`Q`** or **`ESC`** to exit cleanly.
- Violations are automatically saved to `safety_report.csv`.

---

## 📊 Automated Report (`safety_report.csv`)

Every CRITICAL event is instantly logged:

| Timestamp | Workers | Helmets | Vests | Violations | Risk Status |
|-----------|---------|---------|-------|------------|-------------|
| 2026-05-05 10:24:36 | 2 | 1 | 2 | 1 | CRITICAL |

---

## 🛠️ Requirements

```
ultralytics
opencv-python
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 💡 Business Value

| Benefit | Impact |
|---------|--------|
| ✅ Replaces manual safety inspectors | Reduces 24/7 labor costs |
| ✅ Instant violation detection | Prevents accidents before they happen |
| ✅ Automated compliance reports | Eliminates manual paperwork |
| ✅ Runs 100% locally | Zero cloud costs, full data privacy |
| ✅ Dual-model pipeline | Maximum accuracy for both persons and PPE |
| ✅ Works on standard laptops | No specialized hardware needed |

---

## 👤 Authors

**Aman Saeed Mohamed** — AI Developer & Data Analyst  
**Mustapha Ali Gumel** — Team Member  
**Mentor: Akile ODAY**

🔗 [GitHub](https://github.com/Aman-saeed-mohamed) · [LinkedIn](https://www.linkedin.com/in/aman-saeed-mo/)

---

## 📚 References

- Jocher, G., et al. (2023). *Ultralytics YOLOv8*. [github.com/ultralytics](https://github.com/ultralytics/ultralytics)
- Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal.
- OSHA (2022). *Personal Protective Equipment Guidelines*. [osha.gov](https://www.osha.gov/personal-protective-equipment)

---

*📌 This project is developed for academic and research purposes at Eastern Mediterranean University.*