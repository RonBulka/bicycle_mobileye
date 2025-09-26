# Bicycle Safety Vehicle Detection System

A comprehensive computer vision system for bicycle safety that detects and tracks approaching vehicles, providing real-time collision warnings to cyclists. **The project's end goal is to deploy the trained model on a Raspberry Pi with the OAK-D Lite camera for edge computing applications.**

## 🚴‍♂️ Project Overview

This project implements an intelligent vehicle detection and tracking system designed specifically for bicycle safety applications. Using state-of-the-art YOLO object detection and advanced tracking algorithms, it provides real-time warnings about vehicles that pose collision risks to cyclists. The system is optimized for deployment on edge devices, particularly the Raspberry Pi with OAK-D Lite camera for real-world bicycle safety applications.

## ✨ Key Features

- **Real-time Vehicle Detection**: Uses YOLO models (v8, v10, v11) for accurate vehicle detection
- **Advanced Vehicle Tracking**: Kalman filter-based tracking with unique vehicle IDs
- **Collision Warning System**: Time-to-collision (TTC) calculation and audio warnings
- **Edge Computing Deployment**: Optimized for Raspberry Pi + OAK-D Lite deployment
- **Custom Training Pipeline**: Complete dataset preparation and model training workflow
- **Audio Integration**: Real-time audio warnings for dangerous vehicles
- **Performance Optimization**: Edge computing support for real-time processing

## 🎯 End Goal: Raspberry Pi + OAK-D Lite Deployment

The primary objective of this project is to create a portable, battery-powered bicycle safety system that can be mounted on a bicycle and provide real-time vehicle detection and collision warnings. The system will be deployed on:

- **Raspberry Pi 4**: Main processing unit
- **OAK-D Lite Camera**: Stereo depth camera for 3D perception
- **Portable Speaker**: Audio warning system
- **Battery Pack**: Power supply for mobile operation

### Deployment Benefits

- **Portable**: Lightweight system that can be mounted on any bicycle
- **Real-time Processing**: Edge computing eliminates cloud dependency
- **3D Perception**: OAK-D Lite provides depth information for better collision prediction

## 🏗️ System Architecture

```
OAK-D Lite Camera → Raspberry Pi → YOLO Detection → Vehicle Tracking → Collision Analysis → Audio Warning
       ↓                ↓                ↓                 ↓                  ↓                   ↓
   RGB + Depth    Edge Processing  Object Detection   Kalman Filter     TTC Calculation     Speaker Output
```

## 📁 Project Structure

```
bicycle_mobileye/
├── src/                           # Source code
│   ├── vehicle_tracker.py         # Core tracking system
│   ├── train_yolo.py              # Model training script
│   ├── deploy_model.py            # OAK-D deployment
│   ├── convert_model_to_blob.py   # Model conversion for OAK-D
│   ├── downloader.py              # Dataset downloader
│   ├── annotate_images.py         # Image annotation utilities
│   └── constants.py               # Configuration constants
├── tests/                         # Test files
│   ├── predict.py                 # Video prediction script
│   ├── predict_pretrained.py      # Pre-trained model prediction
│   ├── test_vehicle_tracker.py    # Vehicle tracker tests
│   ├── audio_test.py              # Audio system tests
│   └── speaker_test.py            # Speaker functionality tests
├── scripts/                       # Utility scripts
│   ├── check_gpu.py               # GPU availability check
│   ├── preview_video.py           # Video preview utility
│   └── create_images_list.py      # Dataset preparation
├── dataset/                       # Dataset and configuration
│   ├── config.yaml                # Dataset configuration
│   ├── images/                    # Training and validation images
│   └── labels/                    # Annotation labels
├── runs/                          # Training outputs and results
├── evaluation_vids/               # Test videos
│   ├── input/                     # Input test videos
│   └── output/                    # Processed output videos
├── audio/                         # Warning sound files
├── luxonis_output/                # OAK-D model files
│   ├── last.blob                  # Compiled model for OAK-D
│   ├── last.xml                   # OpenVINO XML
│   ├── last.bin                   # OpenVINO binary
│   └── last.json                  # Model metadata
├── docs/                          # Documentation
├── vid_result/                    # Video processing results
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for training)
- **Raspberry Pi 4** (for deployment)
- **OAK-D Lite camera** (for deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bicycle_mobileye.git
   cd bicycle_mobileye
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset**
   ```bash
   python src/downloader.py --export_labels --train_samples 6000 --val_samples 1500
   ```

4. **Train model**
   ```bash
   python src/train_yolo.py --epochs 200 --batch 32
   ```

5. **Test prediction**
   ```bash
   python tests/predict.py --input_name input1.mp4 --output_name output1.mp4
   ```

6. **Deploy model**
   ```bash
   python src/deploy_model.py
   ```

## 🔌 Hardware Deployment

### OAK-D Lite Camera Setup

The OAK-D Lite is a stereo depth camera that provides both RGB and depth information, making it ideal for bicycle safety applications:

```bash
# Convert trained model to blob format for OAK-D
python src/convert_model_to_blob.py

# Test OAK-D deployment
python src/deploy_model.py --test
```

### Raspberry Pi Deployment

The system is optimized for Raspberry Pi deployment with the following components:

#### Hardware Requirements
- **Raspberry Pi 4** (4GB RAM recommended)
- **OAK-D Lite camera**
- **bluetooth speaker/earbuds**
- **Power bank**
- **MicroSD card**

#### Software Setup

```bash
# Install packages on Raspberry Pi
pip install -r requirements.txt

# Test audio system
python tests/audio_test.py

```
#### Manual Installation

##### 1. Install System Dependencies

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git
```

##### 2. Install Python Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

##### 3. Set Up Systemd Service

Copy the service file:
```bash
sudo cp bicycle-mobileye.service /etc/systemd/system/
```

Edit the service file to match your paths:
```bash
sudo nano /etc/systemd/system/bicycle-mobileye.service
```

Enable and start the service:
```bash
# Enable and start
sudo systemctl enable bicycle-mobileye
sudo systemctl start bicycle-mobileye
```
#### Monitoring

##### Service Status
```bash
# Check service status
sudo systemctl status bicycle-mobileye

# Check if running
sudo systemctl is-active bicycle-mobileye

# View recent logs
sudo journalctl -u bicycle-mobileye -n 50

# Follow logs in real-time
sudo journalctl -u bicycle-mobileye -f

# Check startup script
cat /tmp/bicycle_startup.log
```

##### Hardware Status
```bash
# Check camera
lsusb | grep 03e7

# Check Bluetooth
bluetoothctl show

# Check audio
pactl list short sinks
```

## 📊 Model Training

### Dataset Preparation

The system uses the Open Images V7 dataset with custom vehicle annotations:

```bash
# Download and prepare dataset
python src/downloader.py --export_labels

```

### Training Configuration

Key training parameters in `src/constants.py`:
- `EPOCHS = 200`: Training epochs
- `BATCH_SIZE = 32`: Batch size
- `IMAGE_SIZE = 640`: Input image size
- `TRAIN_SAMPLES = 6000`: Training samples
- `VAL_SAMPLES = 1500`: Validation samples

### Model Training

```bash
# Train with custom parameters
python src/train_yolo.py --epochs 200 --batch 32 --imgsz 640

# Train with different model
python src/train_yolo.py --model yolov10n.pt --epochs 150
```

## 🎯 Vehicle Tracking

### Core Components

- **TrackedVehicle**: Individual vehicle tracking with state management
- **VehicleTracker**: Multi-object tracking system
- **WarningStateManager**: Audio warning system
- **Kalman Filter**: State estimation and prediction

### Key Features

- **Multi-object Tracking**: Unique IDs for each detected vehicle
- **Collision Prediction**: Time-to-collision calculation using depth data
- **Speed Estimation**: Based on vehicle size changes
- **Audio Warnings**: Real-time alerts for dangerous vehicles
- **ROI Filtering**: Region-of-interest filtering for side vehicles
- **3D Perception**: Depth-aware collision prediction (OAK-D Lite)

## 🔧 Configuration

### Tracking Parameters

Edit `src/constants.py` to adjust tracking behavior:

```python
CONFIDENCE_THRESHOLD = 0.7      # Detection confidence
SPEED_THRESHOLD = 5.0           # Speed warning threshold
TTC_THRESHOLD = 2.0             # Time-to-collision threshold
IOU_THRESHOLD = 0.3             # Track matching threshold
```

### Kalman Filter Tuning

The system uses an 8-state Kalman filter for robust tracking:

```python
# State: [x, y, w, h, dx, dy, dw, dh]
# x, y: position, w, h: size, dx, dy, dw, dh: velocities
```

## 🎥 Video Processing

### Input/Output

```bash
# Process single video
python tests/predict.py --input_name video.mp4 --output_name output.mp4

```

## 📈 Performance

### Model Performance

- **YOLOv8n**: Fast inference, good accuracy
- **YOLOv10n**: Improved accuracy, moderate speed
- **YOLOv11n**: Best accuracy, slower inference

### Edge Computing Performance

- **Raspberry Pi 4**: 15-25 FPS with optimized models
- **OAK-D Lite**: Real-time depth processing

### Tracking Performance

- **Real-time Processing**: 15-25 FPS on Raspberry Pi
- **Multi-object Tracking**: Handling multiple vehicles

## 🛠️ Development

### Code Structure

- **Modular Design**: Separate modules for different functionalities
- **Type Hints**: Full type annotation for better code quality
- **Error Handling**: Comprehensive error handling and logging
- **Documentation**: Detailed docstrings and comments

### Testing

```bash
# GPU availability check
python scripts/check_gpu.py

# Audio system test
python tests/audio_test.py

# OAK-D Lite test
python src/deploy_model.py --test
```

## 🚴‍♂️ Bicycle Integration

### Mounting System
- **Seat Post Mount**: Main mounting location

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLO model implementations
- **Open Images**: Dataset source
- **DepthAI**: OAK-D camera support
- **OpenCV**: Computer vision library
- **Raspberry Pi Foundation**: Edge computing platform

---

**Safety Notice**: This system is designed to assist cyclists but should not replace proper road safety practices. Always follow traffic laws and use appropriate safety equipment. The system is intended as a supplementary safety aid and should not be relied upon as the primary safety mechanism.