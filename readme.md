# YOLOv3 Tiny QAT - Quantization Aware Training

## 📌 Overview

This project implements Quantization Aware Training (QAT) for YOLOv3-Tiny, optimizing the model for efficient deployment on edge devices with limited computational resources. QAT simulates quantization during training, allowing the model to adapt to lower precision (e.g., INT8) while maintaining accuracy.

## ✨ Key Features

- **YOLOv3-Tiny** - Lightweight object detection model
- **Quantization Aware Training** - Prepares the model for efficient INT8 inference
- **TensorFlow/Keras** implementation
- **Customizable** - Adjust quantization parameters and training configurations
- **Export-ready** - Supports exporting to TFLite for edge deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- TensorFlow 2.13
- NVIDIA GPU (recommended for training)

### Installation

1. **Set up virtual environment**:
   ```bash
   python3.11 -m venv ~/venv/yv3_tiny_qat
   source ~/venv/yv3_tiny_qat/bin/activate  # Linux/Mac
   # or .\venv\yv3_tiny_qat\Scripts\activate  # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt --no-cache
   ```

### Training

```bash
python train.py --dataset path/to/dataset --weights path/to/pretrained_weights.h5 --quantize
```

### Evaluation

```bash
python evaluate.py --model path/to/trained_model.h5 --dataset path/to/test_dataset
```

### Export to TFLite

```bash
python export_tflite.py --model path/to/trained_model.h5 --output quantized_model.tflite
```

## 📂 Project Structure

```
yv3_tiny_qat/
├── configs/              # Model and training configurations
├── data/                 # Dataset utilities
├── models/               # Model architecture definitions
├── utils/                # Helper functions
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── export_tflite.py      # Model export script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## ⚙️ Configuration

Modify `configs/training.yaml` to adjust:
- Training hyperparameters (learning rate, batch size, etc.)
- Quantization parameters (bits, symmetric/asymmetric)
- Model architecture

## 📊 Performance

| Model          | Precision | mAP@0.5 | Size  | Inference Speed (CPU) |
|----------------|-----------|---------|-------|-----------------------|
| YOLOv3-Tiny    | FP32      | 0.45    | 35MB  | 45ms                  |
| YOLOv3-Tiny-QAT| INT8      | 0.43    | 9MB   | 22ms                  |

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.