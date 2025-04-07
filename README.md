# ImgVid_Final_Project

# TinyML Person Detection & Multi-Attribute Recognition on ESP32-S3

**TinyML Person Detection and Attribute Recognition on ESP32-S3 Using the Wake Vision Dataset**  
This project develops two TinyML models – one for binary person detection and one for multi-attribute recognition – trained on the [Wake Vision dataset](https://blog.tensorflow.org/2024/12/introducing-wake-vision-new-dataset-for-person-detection-in-tinyml.html). The optimized models are deployed on the Seeed Studio ESP32-S3 Sense board using TensorFlow Lite for Microcontrollers (TFLM).

---

## Table of Contents

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
  - [Binary Person Detection](#binary-person-detection)
  - [Multi-Attribute Recognition](#multi-attribute-recognition)
- [Model Optimization](#model-optimization)
- [Deployment on ESP32-S3](#deployment-on-esp32-s3)
- [Setup & Installation](#setup--installation)
- [Contributing](#contributing)
- [Resources & References](#resources--references)
- [Project Team & Contacts](#project-team--contacts)
---

## Project Overview

This project aims to push the boundaries of TinyML by:
- **Training:** Developing efficient CNN architectures (e.g., MobileNetV2, MCUNet, MicroNets) using the Wake Vision dataset for both binary person detection and fine-grained attribute recognition.
- **Optimizing:** Applying TensorFlow Model Optimization techniques including Post-Training Quantization (PTQ), Quantization-Aware Training (QAT), pruning, and weight clustering.
- **Deploying:** Converting optimized models to TensorFlow Lite format and deploying them on the Seeed Studio ESP32-S3 Sense board (with an OV2640 camera) for on-device inference.

---

## Repository Structure

```plaintext
tinyml-wakevision-project/
├── README.md                   <- This file
├── data/                       <- Scripts and notes for dataset preparation and augmentation
│   ├── download_wakevision.py
│   └── augmentations.py
├── training/                   <- Python code for model training and evaluation
│   ├── wakevision_person_detection.ipynb
│   ├── wakevision_multi_attribute.ipynb
│   └── export_tflite.py
├── models/                     <- Saved model artifacts
│   ├── person_detect_float.tflite
│   ├── person_detect_int8.tflite
│   └── person_detect_model.h
├── hardware/                    <- Embedded code for deployment
│   ├── arduino_sketch.ino
│   ├── model_data.h

└── docs/                       <- Additional documentation and guides
    └── setup_guide.md


