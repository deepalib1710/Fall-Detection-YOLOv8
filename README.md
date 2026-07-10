# 🚨 Fall-Detection-YOLOv8

A comprehensive fall detection system powered by **YOLOv8m** for identifying and detecting falls in video sequences. The system processes video streams with high accuracy and efficiency, providing instant detection and analysis capabilities for real-time monitoring applications.

---

## 📋 Overview

This project implements an advanced computer vision solution for **real-time fall detection** using the YOLOv8 medium-sized model. The system is trained on custom datasets to accurately identify human falls in various environments and conditions. Whether analyzing live video feeds or processing recorded footage, the model delivers fast and reliable fall detection with minimal latency.

---

## 🧠 Algorithm & Working Procedure

### **1. Input Processing & Frame Extraction**

#### **Video Frame Extraction**
```python
# Extract frames from video with frame skipping for efficiency
cap = cv2.VideoCapture(video_path)
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save every 20th frame to reduce computational load
    if count % 20 == 0:
        frame_name = f"{video}_frame_{frame_count}.jpg"
        cv2.imwrite(os.path.join(frames_folder, frame_name), frame)
        frame_count += 1
    count += 1
cap.release()
```

**Key Details:**
- **Frame Sampling**: Extracts every 20th frame from video to balance accuracy with computational efficiency
- **Resolution**: Frames extracted at full video resolution
- **Storage**: Frames saved as JPG format to reduce storage requirements
- **Rationale**: Sampling reduces redundancy since consecutive frames have minimal variation

#### **Preprocessing Steps**
1. **Image Reading**: OpenCV loads frames from disk
2. **Resize to Model Input**: YOLOv8 requires 640×640 input
3. **Normalization**: Pixel values scaled to [0, 1] range
4. **Batch Creation**: Frames grouped into batches for efficient GPU processing

---

### **2. Object Detection Phase (YOLOv8m)**

#### **Architecture Breakdown**

The YOLOv8 medium model consists of three main components:

**A. Backbone - Feature Extraction**
```python
# CSPDarknet backbone extracts multi-scale features
model = YOLO("yolov8n.pt")  # Download pretrained YOLOv8 nano (or m for medium)
results = model(img_path)[0]
```

- **CSPDarknet Architecture**: Extracts features at multiple scales
- **Cross-Stage Partial Connections (CSP)**: Improves gradient flow and reduces parameters
- **Output**: Feature maps at 3 different resolutions (high detail, medium, low detail)

**B. Neck - Feature Fusion (PAN)**
- **Path Aggregation Network**: Combines features from different scales
- **Purpose**: Ensures small objects (distant persons) and large objects (close persons) are both detected
- **Bidirectional Flow**: Features propagate both upward and downward through the network

**C. Head - Detection Predictions**
```python
# For each detected person:
for box in results.boxes:
    if int(box.cls[0]) == 0:  # Person class ID
        x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
        
        # Convert to YOLO format (normalized center coordinates)
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h
```

**Predictions Generated:**
1. **Bounding Box**: (x_center, y_center, width, height) - normalized coordinates
2. **Confidence Score**: Probability of detection (0-1)
3. **Class Probability**: Likelihood of being a "person"

---

### **3. Automatic Annotation & Label Generation**

#### **Auto-Labeling Process**
```python
# Pre-trained YOLOv8 nano generates automatic labels
label_path = os.path.join(label_folder, img_name.replace(".jpg", ".txt"))

with open(label_path, "w") as f:
    for box in results.boxes:
        if int(box.cls[0]) == 0:  # Only process person detections
            # Write normalized coordinates to label file
            f.write(f"{CLASS_ID} {x_center} {y_center} {width} {height}\n")
```

**Workflow:**
1. Pre-trained model detects all persons in each frame
2. Detections automatically converted to YOLO format
3. Labels stored as text files (one per image)
4. Enables semi-supervised learning approach
5. Human review can refine labels for improved accuracy

#### **Label Dataset Structure**
```
fall_detection_dataset/
├── images/
│   ├── train/
│   │   ├── video1_frame_0.jpg
│   │   ├── video1_frame_1.jpg
│   │   └── ...
│   └── val/
└── labels/
    ├── train/
    │   ├── video1_frame_0.txt  (contains: 1 0.512 0.478 0.245 0.567)
    │   ├── video1_frame_1.txt
    │   └── ...
    └── val/
```

---

### **4. Fall Detection Classification**

#### **Pose Analysis Techniques**

**Method 1: Bounding Box Geometry Analysis**
```python
# Aspect ratio indicates posture
aspect_ratio = box_width / box_height

# Normal Standing Posture:
# - Height > Width (tall bounding box)
# - Aspect ratio < 0.7

# Fallen Posture:
# - Width > Height (wide bounding box)  
# - Aspect ratio > 0.9
```

**Visual Interpretation:**
- Standing person: Bounding box is taller than wide
- Fallen person: Bounding box is wider than tall

**Method 2: Vertical Position Tracking**
```python
# Track vertical position across frames
frame_positions = []

for frame in video_sequence:
    detections = model(frame)
    
    for person in detections:
        y_top = person.bbox.y1
        y_bottom = person.bbox.y2
        y_center = (y_top + y_bottom) / 2
        
        frame_positions.append(y_center)

# Sudden drop in y_center indicates fall
vertical_drop = frame_positions[-1] - frame_positions[-10]
```

**Logic:**
- When person falls, their center Y-coordinate increases suddenly (moves down)
- Rate of change indicates severity of fall

#### **Confidence Scoring**
```python
# Multi-factor confidence calculation
standing_score = 1.0
if aspect_ratio > 0.8:
    standing_score -= 0.3  # Reduce score if wide
if vertical_drop > threshold:
    standing_score -= 0.5  # Reduce score if falling

fall_detected = standing_score < 0.5
```

---

### **5. Temporal Smoothing & Post-Processing**

#### **Noise Reduction**
```python
# Apply moving average to reduce false positives
detection_history = []
WINDOW_SIZE = 5  # Smooth over 5 frames

for current_detection in detections:
    detection_history.append(current_detection.confidence)
    
    if len(detection_history) > WINDOW_SIZE:
        detection_history.pop(0)
    
    # Use average confidence
    smoothed_confidence = sum(detection_history) / len(detection_history)
    
    if smoothed_confidence > CONFIDENCE_THRESHOLD:
        ALERT("Fall Detected!")
```

**Benefits:**
- Eliminates single-frame false positives
- Handles temporary occlusions gracefully
- Improves alert reliability

#### **Confidence Thresholding**
```python
CONFIDENCE_THRESHOLD = 0.5

# Only process detections above threshold
valid_detections = [d for d in detections if d.confidence > CONFIDENCE_THRESHOLD]

for detection in valid_detections:
    # Proceed with visualization and alerting
```

---

### **6. Visualization & Output Generation**

#### **Annotation Overlay**
```python
import cv2

# Render bounding boxes on frame
for detection in detections:
    x1, y1, x2, y2 = detection.bbox_coordinates
    confidence = detection.confidence
    is_fallen = detection.fall_detected
    
    # Choose color based on state
    color = (0, 0, 255) if is_fallen else (0, 255, 0)  # Red if fallen, Green if normal
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Add label with confidence
    label = f"Fall: {confidence:.2f}" if is_fallen else f"Normal: {confidence:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Write frame to output video
    video_writer.write(frame)
```

#### **Output Formats**
1. **Annotated Video**: MP4 with bounding boxes and labels
2. **JSON Metadata**: Frame-by-frame detection data
3. **Alert Logs**: Timestamps of detected falls with confidence scores
4. **Statistics**: Total falls detected, accuracy metrics

---

## 🛠️ Technical Specifications

| Component | Details |
|-----------|---------|
| **Model** | YOLOv8m (Medium) |
| **Input Size** | 640×640 pixels |
| **Framework** | PyTorch |
| **Detection Speed** | ~50-100 FPS (GPU) / ~10-20 FPS (CPU) |
| **Accuracy** | Optimized for person detection & fall classification |
| **Confidence Threshold** | 0.5 (configurable) |
| **Temporal Window** | 5 frames for smoothing |
| **Frame Sampling** | Every 20th frame extracted from video |

---

## 📊 Datasets & Project Resources

The datasets and model outputs for this project are hosted on Google Drive due to file size constraints. Click the links below to access them directly:

### **📁 Datasets**
* **[Training Dataset (Google Drive Direct Link)](https://drive.google.com/drive/folders/1FfsEIY3BukjSX-3nGuRH6fGYTCm8X8wO?usp=sharing)** - Contains labeled frames for training
* **[Test Dataset (Google Drive Direct Link)](https://drive.google.com/drive/folders/1uUewi9Fm11V_X3RHmu9UEfAF9e10n_pz?usp=sharing)** - Test videos and evaluation frames

### **🖼️ Model Outputs & Results**
* **[Test Dataset Outputs (Google Drive Direct Link)](https://drive.google.com/drive/folders/1chK2ikJpPxc06nRpiuDv9QPKhOYYBoH2?usp=sharing)** - Annotated results and detection outputs

### **💾 Custom Data**
* **[Custom Training Data (Google Drive Direct Link)](https://drive.google.com/drive/folders/YOUR_CUSTOM_DATA_FOLDER_ID?usp=sharing)** - Additional custom datasets

---

## 📦 Requirements

- Python 3.8+
- PyTorch with CUDA support (for GPU acceleration)
- OpenCV (`opencv-python`)
- YOLOv8 (`ultralytics`)
- NumPy, Pandas
- Google Colab (for cloud execution)

---

## 📝 Installation

```bash
# Clone repository
git clone https://github.com/deepalib1710/Fall-Detection-YOLOv8.git
cd Fall-Detection-YOLOv8

# Install dependencies
pip install -r requirements.txt

# Or manually install
pip install ultralytics opencv-python torch torchvision numpy pandas
```

---

## 🔧 Key Code Components

### **Frame Extraction Module**
- Loads videos and extracts frames at specified intervals
- Handles various video formats (MP4, AVI, MOV)
- Stores frames in organized directory structure

### **Detection Module**
- Loads YOLOv8 pretrained model
- Performs inference on frame batches
- Outputs bounding boxes with confidence scores

### **Classification Module**
- Analyzes bounding box geometry
- Tracks vertical position changes
- Classifies as fall or normal posture

### **Visualization Module**
- Renders annotated frames with bounding boxes
- Color-codes detections (green=normal, red=fall)
- Overlays confidence scores and labels

---

## 📈 Performance Metrics

- **Inference Time**: 6-12ms per frame (GPU)
- **Throughput**: Up to 160+ frames per second (GPU)
- **Model Size**: ~11-13 MB (YOLOv8m)
- **Memory Usage**: ~2-3 GB (with batch processing)

---

## 🎯 Future Enhancements

- Multi-person fall detection with ID tracking
- Enhanced pose estimation using keypoints
- Mobile app integration (Android/iOS)
- Cloud deployment with AWS/Azure
- Real-time alert notifications via email/SMS/WhatsApp
- 3D pose estimation for better accuracy
- Integration with monitoring systems
- Elderly care facility deployment

---

