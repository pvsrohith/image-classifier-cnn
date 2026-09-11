# image-classifier-cnn

A convolutional neural network image classifier built with PyTorch, trained on the CIFAR-10 dataset. Includes training, evaluation, and single-image inference scripts.

## Features

- Custom 3-layer CNN architecture with pooling and dropout regularization
- Data augmentation (random crop, horizontal flip) for improved generalization
- Training loop with per-epoch test accuracy reporting
- Standalone prediction script for classifying a single image

## Tech Stack

Python, PyTorch, torchvision, Pillow

## Getting Started

```bash
pip install -r requirements.txt
python train.py
```

The CIFAR-10 dataset downloads automatically on first run. After training, model weights are saved to `cnn_cifar10.pth`.

## Predicting on a New Image

```bash
python predict.py path/to/image.jpg
```

## Classes

plane, car, bird, cat, deer, dog, frog, horse, ship, truck
