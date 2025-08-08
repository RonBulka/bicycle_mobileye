# Bicycle Safety Vehicle Detection System

A comprehensive computer vision system for bicycle safety that detects and tracks approaching vehicles, providing real-time collision warnings to cyclists.

## 🚴‍♂️ Project Overview

This project implements an intelligent vehicle detection and tracking system designed specifically for bicycle safety applications. Using state-of-the-art YOLO object detection and advanced tracking algorithms, it provides real-time warnings about vehicles that pose collision risks to cyclists.

## ✨ Key Features

- **Real-time Vehicle Detection**: Uses YOLO models (v8, v10, v11) for accurate vehicle detection
- **Advanced Vehicle Tracking**: Kalman filter-based tracking with unique vehicle IDs
- **Collision Warning System**: Time-to-collision (TTC) calculation and audio warnings
- **Multi-platform Support**: Desktop, OAK-D Lite camera, and Raspberry Pi deployment
- **Custom Training Pipeline**: Complete dataset preparation and model training workflow
- **Audio Integration**: Real-time audio warnings for dangerous vehicles
- **Performance Optimization**: Edge computing support for real-time processing

## 🏗️ System Architecture

```
Video Input → YOLO Detection → Vehicle Tracking → Collision Analysis → Audio Warning
     ↓              ↓                ↓                ↓              ↓
  Camera/File   Object Detection  Kalman Filter   TTC Calculation  Speaker Output
```

## 📁 Project Structure

```
bicycle_mobileye/
├── src/                    # Source code
│   ├── vehicle_tracker.py  # Core tracking system
│   ├── predict.py          # Video prediction script
│   ├── train_yolo.py       # Model training script
│   ├── deploy_model.py     # OAK-D deployment
│   ├── constants.py        # Configuration constants
│   └── ...                 # Additional utilities
├── data/                   # Dataset and configuration
├── runs/                   # Training outputs
├── evaluation_vids/        # Test videos
├── audio/                  # Warning sound files
└── docs/                   # Documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for training)
- OAK-D Lite camera (for edge deployment)

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

5. **Run prediction**
   ```bash
   python src/predict.py --input_name input1.mp4 --output_name output1.mp4
   ```

## 📊 Model Training

### Dataset Preparation

The system uses the Open Images V7 dataset with custom vehicle annotations:

```bash
# Download and prepare dataset
python src/downloader.py --export_labels

# Create image lists for training
python src/create_images_list.py --input-dir data/images/train --output-type train --output-dir data
python src/create_images_list.py --input-dir data/images/val --output-type valid --output-dir data
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
- **Collision Prediction**: Time-to-collision calculation
- **Speed Estimation**: Based on vehicle size changes
- **Audio Warnings**: Real-time alerts for dangerous vehicles
- **ROI Filtering**: Region-of-interest filtering for side vehicles

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
python src/predict.py --input_name video.mp4 --output_name output.mp4

# Batch processing
python src/predict.py --input_dir ./videos/ --output_dir ./results/

# Test mode with visualization
python src/predict.py --test --play
```

### Video Controls

- **Playback**: Interactive video player with controls
- **Frame Capture**: Save specific frames for analysis
- **Performance Metrics**: FPS and processing statistics

## 🔌 Hardware Deployment

### OAK-D Lite Camera

```bash
# Convert model to blob format
python src/convert_model_to_blob.py

# Deploy on OAK-D Lite
python src/deploy_model.py --test
```

### Raspberry Pi

```bash
# Test speaker functionality
python src/speaker_test.py

# Audio system test
python src/audio_test.py
```

## 📈 Performance

### Model Performance

- **YOLOv8n**: Fast inference, good accuracy
- **YOLOv10n**: Improved accuracy, moderate speed
- **YOLOv11n**: Best accuracy, slower inference

### Tracking Performance

- **Real-time Processing**: 30+ FPS on modern hardware
- **Multi-object Tracking**: Up to 10 vehicles simultaneously
- **Low Latency**: <100ms warning response time

## 🛠️ Development

### Code Structure

- **Modular Design**: Separate modules for different functionalities
- **Type Hints**: Full type annotation for better code quality
- **Error Handling**: Comprehensive error handling and logging
- **Documentation**: Detailed docstrings and comments

### Testing

```bash
# GPU availability check
python src/check_gpu.py

# Audio system test
python src/audio_test.py

# Video preview
python src/preview_video.py --video test.mp4
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLO model implementations
- **Open Images**: Dataset source
- **DepthAI**: OAK-D camera support
- **OpenCV**: Computer vision library

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the training logs in `runs/`

---

**Safety Notice**: This system is designed to assist cyclists but should not replace proper road safety practices. Always follow traffic laws and use appropriate safety equipment.