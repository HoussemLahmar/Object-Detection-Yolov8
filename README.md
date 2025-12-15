# Object Detection with Edge AI 🚀

Object Detection project for **Edge AI** based on **YOLOv8**, optimized and deployed on **NVIDIA Jetson Nano** using **TensorRT** for real-time performance.

---

## 📌 Project Overview

This project demonstrates that **state-of-the-art object detection** can run **in real time** on **low-power embedded hardware**. Instead of relying on cloud-based AI, the model runs **directly on the edge**, ensuring:

* Low latency
* Offline operation
* Reduced bandwidth usage
* Better data privacy

**Target platform:** NVIDIA Jetson Nano (4GB)

**Main application domains:**

* Autonomous vehicles
* Mobile robots
* Smart cities
* Intelligent surveillance

---

## 🎯 Objectives

* Deploy **YOLOv8** on an embedded edge device
* Optimize inference using **TensorRT (FP16 / INT8)**
* Achieve **real-time object detection** under strict power and memory constraints
* Evaluate detection accuracy and system stability

---

## 🧠 Why Edge AI?

### Limitations of Cloud AI

* Network latency (50–200 ms or more)
* Internet dependency
* Privacy concerns
* High bandwidth cost for video streaming

### Advantages of Edge AI

* End-to-end latency: **10–50 ms per frame**
* Fully offline operation
* Data processed locally
* Only metadata/events sent to the cloud

---

## 🧩 Hardware Platform

**NVIDIA Jetson Nano**

* GPU: 128-core Maxwell (CUDA-enabled)
* RAM: 4 GB unified memory
* Power consumption: ~7–10 W
* Software stack: JetPack, CUDA, cuDNN, TensorRT, OpenCV

A cost-effective and powerful solution for embedded AI.

---

## 🔍 Model: YOLOv8

YOLOv8 is a **single-stage object detector** known for its speed and accuracy.

**Key characteristics:**

* One-pass detection (You Only Look Once)
* Real-time inference
* Global image context awareness

**Input size:** 640×640 or 416×416

**Output:**

* Bounding boxes
* Class labels
* Confidence scores

---

## 🗂 Dataset

The model was trained on the **Self-Driving Cars Dataset (Kaggle)**.

**Classes:**

* Car
* Truck
* Pedestrian
* Cyclist
* Traffic light

The dataset reflects real urban driving scenarios.

---

## ⚙️ Inference & Optimization Pipeline

1. Train YOLOv8 using PyTorch
2. Export model to ONNX
3. Convert to TensorRT engine (FP16 / INT8)
4. Deploy on Jetson Nano
5. Real-time pipeline:

   * Camera capture
   * Pre-processing
   * GPU inference
   * NMS
   * Visualization

---

## 📊 Performance Evaluation

* **FPS stability:** Constant over long execution (tested over 2 minutes)
* **Latency:** Suitable for real-time applications
* **Accuracy metrics:**

  * mAP@0.5
  * Precision
  * Recall

Observations:

* Most confidence scores lie between **0.4 and 0.6**
* Main errors are false negatives in complex scenes

---

## 📁 Repository Structure

```bash
├── objectdetection-yolov8/
│   ├── inference.py
│   ├── utils/
│   └── requirements.txt
├── Jetson Nano optimisation/
│   ├── export_onnx.py
│   ├── build_trt_engine.py
│   └── README.md
├── Edge_AI_Report_final.pdf
├── Edge_AI_Presentation.pdf
└── README.md
```

---

## ▶️ How to Run

### Prerequisites

* NVIDIA Jetson Nano
* JetPack installed
* Python 3.8+
* CUDA & TensorRT

### Installation

```bash
pip install -r requirements.txt
```

### Run Inference

```bash
python inference.py --engine yolov8.trt --source camera
```

---

## 📈 Results

* Real-time object detection achieved on Jetson Nano
* Stable performance under continuous operation
* Edge AI proven as a viable alternative to cloud-only vision systems

---

## 🔮 Future Work

* Add **object tracking** (DeepSORT, ByteTrack)
* Multi-camera fusion
* Explore lighter and quantized models
* Integrate into a full autonomous robot stack
* Extend dataset and object classes

---

## 👨‍🎓 Authors

* **Houssem-eddine Lahmar**
* **Cheikh Brahim Ahmed Jebbe**

**Supervisor:** Dr. Faten Ben Abdallah
**Institution:** National Engineering School of Tunis (ENIT)

---

## ⭐ Acknowledgments

* Ultralytics YOLOv8
* NVIDIA Jetson & TensorRT
* Kaggle Datasets

If you find this project useful, feel free to ⭐ the repository!
