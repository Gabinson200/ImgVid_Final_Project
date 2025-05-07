# ImgVid_Final_Project

# TinyML Person Detection & Multi-Attribute Recognition on ESP32-S3

**TinyML Person Detection and Attribute Recognition on ESP32-S3 Using the Wake Vision Dataset**  
This project develops two TinyML models – one for binary person detection and one for multi-attribute recognition – trained on the [Wake Vision dataset](https://blog.tensorflow.org/2024/12/introducing-wake-vision-new-dataset-for-person-detection-in-tinyml.html). The optimized models are deployed on the Seeed Studio ESP32-S3 Sense board using TensorFlow Lite for Microcontrollers (TFLM).
[main website](https://wakevision.ai/)

---

## Project Overview

This project aims to push the boundaries of TinyML by:
- **Training:** Developing efficient CNN architectures (e.g., MobileNetV2, MCUNet, MicroNets) using the Wake Vision dataset for both binary person detection and fine-grained attribute recognition.
- **Optimizing:** Applying TensorFlow Model Optimization techniques including Post-Training Quantization (PTQ), Quantization-Aware Training (QAT), pruning, and weight clustering.
- **Deploying:** Converting optimized models to TensorFlow Lite format and deploying them on the Seeed Studio ESP32-S3 Sense board (with an OV2640 camera) for on-device inference.

---


