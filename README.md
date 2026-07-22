# Native Raspberry Pi 5 (AI Hat+ 26 TOPS) Face Recognition Application

A 100% native, high-performance, real-time Face Recognition System for **Raspberry Pi 5** powered by the **Raspberry Pi AI Hat+ 26 TOPS** (Hailo-8 accelerator).

Built natively using **HailoRT Python SDK**, **OpenCV**, and **LanceDB**. Zero third-party cloud SDK dependencies or cloud license token requirements.

---

## 🌟 Key Architecture

- **100% Offline & Native**: Runs natively using official `hailo_platform` HailoRT Python SDK on Raspberry Pi 5.
- **Dual Engine Core**:
  - **Hailo-8 NPU Engine**: Runs compiled `.hef` model pipelines directly on the 26 TOPS Hailo-8 accelerator.
  - **Native OpenCV Engine**: High-speed native `FaceDetectorYN` and `FaceRecognizerSF` fallback engines.
- **4-Stage Recognition Pipeline**:
  1. **Face & Keypoint Landmark Detection**: Extracts bounding boxes and 5 keypoints (eyes, nose, mouth corners).
  2. **ArcFace Affine Alignment**: Standard 112x112 facial alignment via `cv2.estimateAffinePartial2D`.
  3. **Feature Embedding Extraction**: ArcFace MobileFaceNet generating 512-dimensional feature vectors.
  4. **LanceDB Vector Search**: Vector database with Cosine similarity metric and threshold matching.
- **Smart Camera Management**: Automatically uses `Picamera2` for Raspberry Pi Camera Module 3 (IMX708 sensor) and OpenCV V4L2 for USB webcams.

---

## 🛠 Hardware & System Requirements

- **Raspberry Pi 5** (4GB / 8GB) running Raspberry Pi OS (64-bit Bookworm).
- **Raspberry Pi AI Hat+ (26 TOPS)** equipped with Hailo-8 chip.
- **Camera**: Raspberry Pi Camera Module 3 or USB Webcam.
- **Hailo Drivers**: Installed Hailo PCIe driver & `hailort` runtime (`sudo apt install -y hailort-drivers hailort`).

---

## 🚀 Installation & Setup

### 1. Enable PCIe on Raspberry Pi 5
Add the following line to `/boot/firmware/config.txt`:
```bash
dtparam=pciex1
dtparam=pciex1_gen=3
```
Reboot:
```bash
sudo reboot
```

### 2. Clone App & Install Dependencies
```bash
cd /Users/pasan/Documents/PasanLive/Upside/Upside Solutions/projects/ML/face-recognition-app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage Guide

### 1. Launch Web UI Dashboard (Recommended)
Start the modern web server dashboard:
```bash
python3 main.py web --port 5000
```
Open your browser at `http://<raspberry-pi-ip>:5000` to view the live camera stream, register new faces, adjust similarity thresholds, and manage enrolled identities.

---

### 2. Command Line Interface (CLI)

#### Enroll Faces into Database
Enroll an entire folder of images (images named like `Alice_1.jpg`, `Bob_2.png`):
```bash
python3 main.py enroll --path ./sample_dataset
```
Or enroll a single face photo:
```bash
python3 main.py enroll --path ./alice.jpg --name Alice
```

#### Run Real-Time Camera Stream
Run real-time video feed window (press `q` to quit):
```bash
python3 main.py run --source 0 --threshold 0.36
```

#### Manage Database
List all enrolled persons:
```bash
python3 main.py list
```

Delete an enrolled identity:
```bash
python3 main.py delete --name Alice
```

---

## 📐 Project Structure

```
face-recognition-app/
├── config.py             # Configuration parameters & model paths
├── main.py               # Native CLI Application entrypoint
├── pipeline.py           # 4-stage Face Recognition Pipeline
├── hailo_engine.py       # Native HailoRT (hailo_platform) HEF runner
├── camera_wrapper.py     # Smart Camera Manager (Picamera2 + OpenCV V4L2)
├── requirements.txt      # Python dependencies (Zero cloud SDKs)
├── README.md             # Complete setup & usage guide
├── face_engine/          # Core AI inference & facial image processing
│   ├── alignment.py      # ArcFace 112x112 affine alignment math
│   ├── detector.py       # Native face & keypoint detector (Hailo / OpenCV)
│   └── embedder.py       # Native face feature embedder (Hailo / OpenCV)
└── database/             # LanceDB Vector Storage
    ├── schema.py         # LanceModel 512-D schema definition
    └── manager.py        # Vector query, enrollment & management logic
```
