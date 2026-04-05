# Traffic Violation Detection System

An AI-powered **real-time traffic monitoring system** that detects **helmet violations** and **triple riding** using computer vision.

---

# Overview

This project uses **deep learning and computer vision** to automatically detect traffic rule violations from live webcam feed.
It identifies **motorcycles, riders**, and checks for:

* 🚨 Triple Riding (more than 2 riders)
* 🪖 Helmet / No Helmet violations (only for riders)

The system also **captures and stores violation images** with timestamps for evidence.

---

# Features

*  Real-time webcam detection
*  Person &  motorcycle detection
*  Triple riding detection using distance logic
*  Helmet / No Helmet detection
*  Smart filtering (detects helmet only for riders)
*  Automatic image saving for violations
*  Cooldown system to avoid duplicate captures
*  FPS display for performance monitoring

---

# Tech Stack

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* NumPy

---

# How It Works

1. Detects **persons and motorcycles** using YOLOv8
2. Calculates distance between riders and bikes
3. If more than 2 persons are near a bike → **Triple Riding Detected**
4. Helmet detection is applied only to riders (not pedestrians)
5. If no helmet is detected → **Violation flagged**
6. Captures and saves images with timestamp

---

# Project Structure

```bash
helmet-triple-detection/
│
├── models/              # Helmet detection model
├── outputs/             # Saved violation images
├── src/
│   └── main.py          # Main detection script
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Installation

```bash
git clone https://github.com/your-username/Helmet-TripleRiding-Detection.git
cd Helmet-TripleRiding-Detection

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---


# Key Highlights

* Real-time AI-based traffic violation detection
* Context-aware helmet detection (only for riders)
* Distance-based rider-bike association
* Automated evidence generation system

---

# Limitations

* Accuracy depends on camera angle and lighting
* Occlusion (overlapping objects) may affect detection
* Distance threshold may need tuning

---

# Future Improvements

*  Number plate detection (OCR)
*  Web dashboard for monitoring (MERN stack)
*  Integration with CCTV cameras
*  Object tracking (assign ID to each rider)

---

# Author

**Pratik Kumar**

---

# Acknowledgements

* YOLOv8 by Ultralytics
* OpenCV community

---
