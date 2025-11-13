# 🎥 Computer Vision Engineer Assessment  
Multi-threaded Video Processing • Queue • FPS Meter • Multi-Camera • YOLOv8 GPU Inference

---

## 📌 Overview

This project demonstrates a complete real-time video processing pipeline using Python.  
It satisfies all requirements of the Computer Vision Engineer assessment:

### ✔ Reads frames from webcam or RTSP  
### ✔ Uses **two threads** (capture thread + processing thread)  
### ✔ Uses a **thread-safe queue**  
### ✔ Displays **FPS** in real time  
### ✔ Simulates or runs **real AI inference** (YOLOv8)  
### ✔ Supports **multiple cameras**  
### ✔ Pressing **'q'** stops everything cleanly  

---

## 📂 Project Files

cv_assessment/
│── main.py # Single-camera threading
│── multi_camera.py # Multi-camera + YOLOv8 GPU inference
│── README.md # Project documentation


---

## ⚙️ Requirements

Install dependencies:

```bash
pip install opencv-python numpy ultralytics
