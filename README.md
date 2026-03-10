# Anemalyze — Non-Invasive Early Anemia Detection System Using Convolutional Neural Network in Women of Reproductive Age

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Flask-API-lightgrey?logo=flask" />
  <img src="https://img.shields.io/badge/Laravel-Frontend-red?logo=laravel" />
  <img src="https://img.shields.io/badge/Raspberry%20Pi-5-C51A4A?logo=raspberrypi" />
  <img src="https://img.shields.io/badge/Accuracy-86%25-brightgreen" />
</p>

> A non-invasive anemia detection system for women of reproductive age, combining conjunctival image analysis with physiological sensor measurements using deep learning.

---

## Overview

Anemalyze detects anemia without blood sampling by analyzing conjunctival (inner eyelid) images using computer vision, measuring heart rate and SpO2 via MAX30100 pulse oximeter sensor, and classifying results using a MobileNetV2 deep learning model.

The system is designed for clinical use, targeting accuracy ≥85% with system feedback ≤5 seconds.

---

## Key Features

- **Non-invasive** — No blood sampling required
- **Real-time detection** — Computation time ≤5 seconds
- **Dual-modality** — Combines image analysis + physiological sensors
- **Web-based interface** — Laravel frontend 
- **Clinical feedback** — LED indicators that on < 5s

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Hardware | Raspberry Pi 5, Pi Camera v3, MAX30100 sensor, LED |
| Segmentation | LinkNet (Segmentation Models PyTorch) |
| Classification | MobileNetV2 + ImageNet preprocessing |
| Backend API | Flask (Python) |
| Frontend | Laravel (PHP) |
| Database | MySQL |

---

## Project Structure

```
anaemia_detection-master/
├── models/
│   ├── __init__.py
│   ├── classification_loader.py   # Load MobileNetV2 model
│   └── segmentation_loader.py     # Load LinkNet model
├── pipeline/
│   ├── __init__.py
│   ├── capture_raspi.py           # Pi Camera v3 capture
│   ├── classification.py          # Anemia classification logic
│   ├── crop.py                    # Conjunctiva region cropping
│   ├── main_pipeline.py           # Main detection pipeline
│   ├── preprocessing.py           # Image preprocessing
│   └── segmentation.py            # Conjunctiva segmentation
├── results/                       # Output results storage
├── utils/
│   ├── __init__.py
│   ├── save_results.py            # Result persistence
│   └── visualization.py          # Result visualization
├── .gitignore
├── api.py                         # Flask API entry point
├── config.py                      # Configuration settings
├── main.py                        # Main application entry
├── max30100.py                    # MAX30100 sensor interface (see Credits)
└── requirements.txt
```

---

## Installation

### Prerequisites
- Raspberry Pi 5 with Raspberry Pi OS
- Python 3.10+
- Pi Camera v3 connected
- MAX30100 sensor connected via I2C

### Steps

```bash
# Clone the repository
git clone https://github.com/arraudhafazyar/anaemia_detection-master.git
cd anaemia_detection-master

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask API
python api.py
```

---

## Hardware Setup

```
Raspberry Pi 5
├── Pi Camera v3         → CSI port
├── MAX30100 sensor      → I2C (SDA: GPIO2, SCL: GPIO3, VIN:3.3 V )
└── LED                  → Anode (+): GPIO26
```

Enable I2C:
```bash
sudo raspi-config
# Interface Options → I2C → Enable
```

---

## Model Architecture

### Segmentation — LinkNet
- Segments the conjunctival region from eye images
- Library: `segmentation-models-pytorch`

### Classification — MobileNetV2
- Input: preprocessed conjunctival ROI
- Pretrained on ImageNet, fine-tuned for anemia classification
- Validation accuracy: **99.7%** (2 misclassifications / 744 samples)
- Training split: 80% train / 20% validation (stratified)

---

## Performance

| Metric | Target | Result |
|--------|--------|--------|
| Classification Accuracy | ≥ 85% | **86%** |
| Computation Time | ≤ 5 seconds | Met |
| System Feedback | ≤ 5 seconds | Met |



## Credits

- `max30100.py` is adapted from [DinMuhammad1994/max30100_with_raspberrypi](https://github.com/DinMuhammad1994/max30100_with_raspberrypi)

---

## License

This project is developed as an undergraduate thesis at **Universitas Andalas**.

---

## Author

**Arraudha Fazya Ramadhani** — Computer Engineering, Universitas Andalas
