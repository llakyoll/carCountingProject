# 🚗 Real-Time Car Counting with YOLO + TensorRT

This project is a **high-performance real-time vehicle counting system** built using **YOLO object detection optimized with TensorRT**.  
It detects, tracks, and counts vehicles based on their movement direction across a predefined line.

---

## 🎯 Project Goal

The goal of this project is to create a **low-latency, high-FPS vehicle counting application** by leveraging GPU acceleration with TensorRT.

Suitable for:

- Smart city solutions  
- Traffic density analysis  
- Parking management  
- Security systems  
- Edge AI deployments (Jetson devices)

---

## ⚙️ Technologies Used

- **Python**
- **YOLO (Object Detection)**
- **TensorRT** → Maximum GPU performance
- **OpenCV** → Video processing
- **NumPy**
- **Object Tracking** → Unique vehicle IDs

---

## 🧠 How It Works

1️⃣ Video stream is captured from a camera or video file.  
2️⃣ TensorRT-optimized YOLO model detects vehicles in each frame.  
3️⃣ A tracking algorithm assigns a unique ID to every vehicle.  
4️⃣ When a vehicle crosses the reference line, its direction is analyzed:

