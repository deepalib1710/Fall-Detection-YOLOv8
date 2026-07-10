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
2. **Resize to Model Input**: Images are configured according to training phases (640×640 standard or 960×960 fine-tuning scale)
3. **Normalization**: Pixel values scaled to [0, 1] range
4. **Batch Creation**: Frames grouped into batches (e.g., batch=4 for high-resolution refinement) for efficient GPU processing

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
3. **Class Probability**: Likelihood of target labels (0: Fall, 1: Normal)

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
4. Algorithmic lookup matches frame filenames against isolated standard datasets to isolate Fall indices (0) vs Normal indices (1)
5. Dataset balancing yields a final training composition split (e.g., 622 Fall frames, 391 Normal frames)

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
box_width = x2 - x1
box_height = y2 - y1
aspect_ratio = box_height / box_width

```

**Visual Interpretation:**
Standing/Normal posture: High vertical box profiles ($\text{height} > \text{width}$)
Fallen posture: Compressed vertical height profiles relative to horizontal expansion ($\frac{\text{height}}{\text{width}} < 0.8$)

**Method 2: Multi-Stage Network Training Optimization**
```python
# Multi-stage custom weights tuning
# Stage 1: Native Baseline Configuration
model.train(data="data.yaml", epochs=40, imgsz=640)  # Reaches ~0.940 mAP50

# Stage 2: High Resolution Fine-Tuning 
model.train(data="data.yaml", epochs=20, imgsz=960, batch=4) # Finetuned weights refinement
```

**Dynamic Risk Scoring Calculation*
```python
# Evaluates relative proximity metrics based on vertical spatial positions
risk_score = 0.0
y_center = (y1 + y2) / 2

if y_center > 400:
    risk_score = 0.9  # Height boundary zone triggers severe fall alerts
elif y_center > 250:
    risk_score = 0.5  # Mid-range spatial warning zone
```
---

### **5. Temporal Smoothing & Post-Processing**

#### **Noise Reduction**
```python
# Inference processes filter out secondary low-confidence noise 
results = model.predict(source=img, conf=0.25)
```

**Benefits:**
- Eliminates low confidence background artifacts
- Maintains high sensitivity toward actual human target shapes
- Structural class mapping logic acts as an extra confirmation filter for edge cases

---

### **6. Visualization & Output Generation**

#### **Annotation Overlay**
```python
# Draw target status identifiers
if cls_id == 0 or aspect_ratio < 0.8:
    label = f"fall {conf:.2f} Risk: {risk_score}"
    color = (0, 0, 255)  # Crimson Alert Red for verified falls
else:
    label = f"normal {conf:.2f}"
    color = (0, 255, 0)  # Bright Green for standing/normal states

cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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
| **Input Size** | Multi-Scale Training Pipeline (640×640 Baseline $\rightarrow$ 960×960 Fine-Tuning) |
| **Framework** | PyTorch |
| **Detection Speed** | ~50-100 FPS (GPU) / ~10-20 FPS (CPU) |
| **Accuracy** | Optimized for person detection & fall classification |
| **Fallback Triggers** |Aspect Ratio Threshold ($\frac{\text{height}}{\text{width}} < 0.8$) + Proximity Risk Scaling |
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

