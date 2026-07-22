# Raspberry Pi 5 (AI Hat+ 26 TOPS) Face Recognition Application

A complete, high-performance, real-time Face Recognition System designed specifically for **Raspberry Pi 5** powered by the **Raspberry Pi AI Hat+ 26 TOPS** (featuring the Hailo-8 AI accelerator chip).

Built following the official [Hailo Community Face Recognition Architecture](https://community.hailo.ai/t/a-comprehensive-guide-to-building-a-face-recognition-system/8803).

---

## 🌟 Key Features

- **26 TOPS Acceleration**: Leverages the Hailo-8 NPU on the Raspberry Pi AI Hat+ for ultra-fast deep learning inference.
- **4-Stage Recognition Pipeline**:
  1. **Face Detection & Keypoint Landmark Extraction**: Uses SCRFD (`scrfd_10g` / `scrfd_2.5g`) or RetinaFace / YOLOv8 face models with 5 facial keypoints (eyes, nose, mouth corners).
  2. **ArcFace Affine Alignment**: Standard 112x112 facial alignment via `cv2.estimateAffinePartial2D`.
  3. **Feature Embedding Extraction**: ArcFace MobileFaceNet generating 512-dimensional feature vectors on Hailo NPU.
  4. **Vector DB Search**: LanceDB vector database with Cosine similarity metric and threshold filtering.
- **Dual User Interfaces**:
  - **Modern Web Dashboard**: Real-time camera feed streaming, identity enrollment modal, vector list management, and similarity threshold adjustment.
  - **Command Line Interface (CLI)**: Headless stream processing, directory batch enrollment, listing, and deletion commands.

---

## 🛠 Hardware & System Requirements

- **Raspberry Pi 5** (4GB / 8GB) running Raspberry Pi OS (64-bit Bookworm).
- **Raspberry Pi AI Hat+ (26 TOPS)** equipped with Hailo-8 chip.
- **Camera**: Raspberry Pi Camera Module 3 or USB Webcam.
- **Hailo Drivers**: Installed Hailo PCIe driver & `hailort` runtime.

---

## 🚀 Installation & Setup

### 1. Enable PCIe on Raspberry Pi 5
Add the following line to `/boot/firmware/config.txt`:
```bash
dtparam=pciex1
dtparam=pciex1_gen=3
```
Reboot your Raspberry Pi 5:
```bash
sudo reboot
```

### 2. Install Hailo Drivers & Software Stack
```bash
sudo apt update
sudo apt install -y hailort-drivers hailort
```
Verify the Hailo-8 device is recognized:
```bash
hailortcli fw-control identify
```

### 3. Clone App & Create Virtual Environment
```bash
cd /Users/pasan/Documents/PasanLive/Upside/Upside Solutions/projects/ML/face-recognition-app

# Create python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. (Optional & Recommended) Download Local Models for Offline Execution
To run offline without requiring a DeGirum cloud token, clone the model repository from `DeGirum/hailo_examples`:
```bash
git clone https://github.com/DeGirum/hailo_examples.git ../hailo_examples
```
*The app automatically detects `./models`, `../hailo_examples/models`, or `~/hailo_examples/models` and runs 100% locally on the Hailo-8 NPU without cloud authentication.*

---

## 💻 Usage Guide

### Option 1: Launch Web UI Dashboard (Recommended)
Start the modern web server dashboard:
```bash
python3 main.py web --port 5000
```
Open your browser at `http://<raspberry-pi-ip>:5000` to view the live camera stream, register new faces, adjust similarity thresholds, and manage enrolled identities.

---

### Option 2: Command Line Interface (CLI)

#### 1. Enroll Faces into Database
Enroll an entire folder of images (images named like `Alice_1.jpg`, `Bob_2.png`):
```bash
python3 main.py enroll --path ./sample_dataset
```
Or enroll a single face photo:
```bash
python3 main.py enroll --path ./alice.jpg --name Alice
```

#### 2. Run Real-Time Camera Recognition Stream
Run real-time video window feed (press `q` to quit):
```bash
python3 main.py run --source 0 --threshold 0.35
```

#### 3. Manage Database
List all enrolled persons:
```bash
python3 main.py list
```

Delete an enrolled identity:
```bash
python3 main.py delete --name Alice
```

---

## ⚙️ Configuration & Environment Variables

Custom settings can be configured in `config.py` or overridden via environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `FACE_APP_TARGET` | `hailo8` | Target hardware chip (`hailo8` for 26 TOPS AI Hat+, `hailo8l` for 13 TOPS) |
| `FACE_APP_THRESHOLD` | `0.35` | Cosine similarity match threshold |
| `FACE_APP_HOST` | `@local` | DeGirum PySDK host (`@local` for local Hailo, `@cloud` for cloud fallback) |
| `FACE_APP_CAM_INDEX` | `0` | Default camera device index |
| `FACE_APP_DB_URI` | `./face_database` | LanceDB storage path |

---

## 📐 Project Structure

```
face-recognition-app/
├── config.py             # System configuration parameters & model selection
├── main.py               # CLI Application entrypoint
├── pipeline.py           # End-to-end 4-stage Face Recognition Pipeline
├── requirements.txt      # Python dependencies
├── README.md             # Complete setup & usage guide
├── face_engine/          # Core AI inference & facial image processing
│   ├── alignment.py      # ArcFace 112x112 affine alignment math
│   ├── detector.py       # Hailo-8 SCRFD face detector wrapper
│   └── embedder.py       # Hailo-8 ArcFace MobileFaceNet embedding wrapper
├── database/             # LanceDB Vector Storage
│   ├── schema.py         # LanceModel 512-D schema definition
│   └── manager.py        # Vector query, enrollment & management logic
└── webui/                # Web Dashboard Server
    ├── app.py            # Flask Web App backend with MJPEG video stream
    └── templates/
        └── index.html    # Modern HTML5/CSS3 Web UI front-end
```
