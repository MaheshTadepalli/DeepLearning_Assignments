# GNR638 – Programming Assignment  
## Custom Deep Learning Framework (C++ Backend + Python Frontend)

---

## 1. Overview

This repository implements a custom deep learning framework from scratch as part of the GNR638: Machine Learning for Remote Sensing - II course.

The framework includes:

- Tensor abstraction with gradient tracking
- Automatic differentiation and backpropagation
- Convolutional layers
- Activation functions
- Pooling layers
- Fully connected layers
- Softmax + Cross Entropy loss
- SGD optimizer with L2 regularization
- C++ backend with Python bindings (via pybind11)

No external deep learning or numerical libraries are used.

---

## 2. Repository Structure

assign1/
│
├── cpp/
│ ├── tensor.h
│ ├── ops.cpp
│ ├── conv.cpp
│ ├── bindings.cpp
│ ├── CMakeLists.txt
│
├── python/
│ ├── dataset.py
│ ├── model.py
│ ├── loss.py
│ ├── optimizer.py
│ ├── train.py
│
└── README.md

yaml
Copy code

---

## 3. Framework Components

### 3.1 Tensor

- Stores data and gradients
- Tracks `requires_grad`
- Supports backpropagation via `backward()`
- Implemented in C++

### 3.2 Implemented Operations

- add
- matmul
- conv2d
- relu
- sigmoid
- maxpool2d
- flatten
- linear
- softmax_cross_entropy (fused)

All operations support automatic differentiation.

---

## 4. Dataset Handling

- Dataset is expected in image-folder format:

root/
class_1/
image1.png
image2.png
class_2/
image3.png
...

yaml
Copy code

- Only PNG images are loaded.
- Images are resized to 32×32.
- Pixel values normalized to [0,1].
- All images are preloaded into memory.
- Dataset loading time is measured and printed.

---

## 5. Model Architecture

### Dataset 1 (10 Classes)

Input: 32x32x3
Conv: 3x3, 3→8
ReLU
MaxPool 2x2
Flatten
FC: 1800→10

shell
Copy code

### Dataset 2 (100 Classes)

Input: 32x32x3
Conv: 3x3, 3→16
ReLU
MaxPool 2x2
Flatten
FC: 3600→100

yaml
Copy code

All layers are implemented using the custom framework.

---

## 6. Training Configuration

Training uses:

- SGD optimizer
- L2 regularization (weight decay)
- Train/Validation split (80/20)
- Early stopping (patience-based)
- Best model weight saving

Hyperparameters are configurable inside `train.py`.

---

## 7. Building the C++ Backend

### Requirements

- Python 3.12
- CMake
- C++ compiler
- pybind11

### Steps

From inside the `cpp` directory:

mkdir build
cd build
cmake ..
cmake --build . --config Release
This generates the Python extension module:

scss
Copy code
mydl_cpp.pyd   (Windows)
mydl_cpp.so    (Linux/Mac)
Make sure this file is accessible from the python/ directory.

8. Running Training
From inside the python directory:

bash
Copy code
python train.py --dataset_path <PATH_TO_DATASET>
Example:

bash
Copy code
python train.py --dataset_path /path/to/dataset1
During training, the script prints:

Dataset size

Dataset loading time

Parameter count

MACs

FLOPs

Training loss and accuracy

Validation loss and accuracy

Epoch time

The best model is saved as:

Copy code
best_model.pkl
9. Running Evaluation
Evaluation script should be run with:

Path to hidden dataset

Path to saved weights (best_model.pkl)

No code modifications are required.

10. Performance & Constraints
Training completes within 3 hours.

Evaluation completes within 1 hour.

No third-party deep learning libraries used.

Only standard library + OpenCV (image loading) used.

All operations implemented from scratch.

11. Model Complexity Reporting
The framework computes:

Total trainable parameters

MACs per forward pass

FLOPs per forward pass

These are printed during training and documented in the report.

12. Reproducibility
Random seed fixed for weight initialization and dataset shuffling.

Deterministic behavior ensured.

13. Sources Used
Official Python documentation

C++ standard library documentation

Pybind11 documentation

Course materials

AI assistance (ChatGPT) for debugging and architectural guidance

14. Honour Code Compliance
This implementation:

Uses only allowed tools.

Does not use any external deep learning frameworks.

Does not use NumPy or similar numerical libraries.

Adheres to all assignment constraints.
