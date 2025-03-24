# Real-Time Object Detection System Using Machine Learning

## Introduction
Welcome to the Git repository for the **Real-Time Object Detection System Using Machine Learning**. This project leverages deep learning and computer vision techniques to enable fast and accurate object detection in real-time video streams. 

This repository serves as a central hub for the code, documentation, and resources related to this system. The goal is to create a robust object detection pipeline suitable for various real-world applications such as **autonomous systems, surveillance, retail, and assistive AI**.

## Project Overview
The **Real-Time Object Detection System** is built using **YOLO (You Only Look Once)**, a state-of-the-art object detection model known for its speed and accuracy. The system is capable of:

- **Detecting multiple objects** in real-time video streams
- **Recognizing and classifying objects** with high precision
- **Extracting and identifying text (OCR)** from images and videos
- **Running efficiently on both edge devices and cloud infrastructure**

The project addresses key challenges in **real-time detection**, **scalability**, and **deployment flexibility**, making it suitable for various industries.

## Getting Started
This section provides instructions on setting up the environment and running the project.

### Installation
#### Using Conda
```bash
# TensorFlow CPU
conda env create -f conda-cpu.yml
conda activate object-detection-cpu

# TensorFlow GPU
conda env create -f conda-gpu.yml
conda activate object-detection-gpu
```

#### Using Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

#### NVIDIA Driver & CUDA Setup (For GPU)
Ensure you have the appropriate **CUDA Toolkit (version 10.1 or later)** installed for TensorFlow GPU compatibility.

## Downloading Pretrained Weights
The system utilizes pre-trained YOLO weights for accurate object detection. Download the official YOLOv4 weights:
```bash
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights -P weights/
```

## Running the Object Detection System
Run the detection script with different configurations:
```bash
python yolo_project.py --weights ./weights/yolov4-416 --model yolov4 --video ./videos/sample.mp4
```
For webcam input:
```bash
python yolo_project.py --weights ./weights/yolov4-416 --model yolov4 --video 0
```
To save the output video:
```bash
python yolo_project.py --weights ./weights/yolov4-416 --model yolov4 --video ./videos/sample.mp4 --output ./detections/output.avi
```

### Command Line Arguments
| Argument        | Description |
|----------------|-------------|
| `--video`      | Path to input video (use `0` for webcam) |
| `--output`     | Path to save output video |
| `--weights`    | Path to YOLO weights file |
| `--model`      | Choose `yolov3` or `yolov4` |
| `--framework`  | Framework to use (`tf`, `trt`, `tflite`) |
| `--info`       | Print detection details (class, confidence, bounding box coordinates) |
| `--count`      | Count total objects detected per class |

## Object Counting Feature
This system includes a feature to **count detected objects** in real-time, either as a total count or per class. 
To enable per-class counting, use:
```bash
python yolo_project.py --weights ./weights/yolov4-416 --model yolov4 --video ./videos/sample.mp4 --count
```

## References
This project is based on various research papers and open-source implementations:
- **YOLOv4: Optimal Speed and Accuracy of Object Detection** ([Paper](https://arxiv.org/abs/2004.10934))
- **Darknet: YOLO Neural Network Implementation** ([GitHub](https://github.com/AlexeyAB/darknet))
- **TensorFlow YOLO Implementations** ([GitHub](https://github.com/hunglc007/tensorflow-yolov4-tflite))

## Contributing
If you'd like to contribute, please create a pull request or report issues.

---
This repository serves as a solid foundation for real-time object detection projects. If you find this useful, feel free to connect and collaborate!
