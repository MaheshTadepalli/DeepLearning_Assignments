##GNR638 – Programming Assignment
Custom Deep Learning Framework (C++ Backend + Python Frontend)
👤 Student Information

Course: Machine Learning for Remote Sensing – II (Spring 2025-26)
Assignment: Design a Deep Learning Framework

📌 Overview

This repository contains a custom deep learning framework built completely from scratch using:

C++ backend (tensor operations, autograd, CNN layers)

Python frontend (training, evaluation, dataset handling)

⚠️ No deep learning frameworks (PyTorch, TensorFlow, NumPy, etc.) were used.
Only standard libraries and OpenCV (for image loading) are used as permitted.

🧠 Framework Features

The framework supports:

Tensor abstraction with gradient tracking

Automatic differentiation (backpropagation)

Convolution layer

ReLU activation

MaxPooling layer

Fully connected layer

Softmax + Cross Entropy loss

SGD optimizer with L2 regularization

Training / Validation split

Early stopping

Model saving & loading

🏗 Model Architecture

The CNN used for both Dataset-1 and Dataset-2:

Input: 32×32×3 image

Conv1: 3×3 kernel, 3 → 16 filters
ReLU
MaxPool (2×2, stride 2)

Conv2: 3×3 kernel, 16 → 32 filters
ReLU
MaxPool (2×2, stride 2)
MaxPool (2×2, stride 2)

Flatten

Fully Connected:
(3×3×32) → num_classes

Softmax (via fused softmax_cross_entropy)

📊 Model Complexity

Printed automatically during training:

Total parameters

MACs per forward pass

FLOPs per forward pass

Example (Dataset-2, 100 classes):

Total parameters ≈ 30,000
Total MACs ≈ 612,288
Total FLOPs ≈ 1,224,576

📂 Dataset Format

Expected structure:

dataset_root/
    class_0/
        img1.png
        img2.png
        ...
    class_1/
        ...


Requirements:

PNG images only

Automatically resized to 32×32

Labels inferred from folder names

Entire dataset preloaded into memory

Dataset loading time printed

⚙️ Installation & Build Instructions
1️⃣ Build C++ Backend

From the cpp/ directory:

mkdir build
cd build
cmake ..
cmake --build . --config Release


This generates the Python module:

mydl_cpp

2️⃣ Install Python Requirements

Only OpenCV is required:

pip install opencv-python

🚀 Training

Edit dataset path in train.py:

train(
    "path_to_dataset",
    epochs=4,
    batch_size=256,
    lr=0.005,
    weight_decay=2e-4,
    patience=1
)


Then run:

python train.py


During training, the script prints:

Dataset loading time

Parameter count

MACs and FLOPs

Training loss

Training accuracy

Validation loss

Validation accuracy

Epoch time

Early stopping status

Best model is saved as:

best_model.pkl

🧪 Evaluation

Edit dataset path and weight path in evaluation.py:

evaluate(
    dataset_path="path_to_test_dataset",
    weights_path="best_model.pkl"
)


Run:

python evaluation.py


Outputs:

Dataset loading time

Parameter count

MACs and FLOPs

Evaluation accuracy

Evaluation time

⚠️ No code modification required by grader — only dataset path and weights path are needed.

💾 Model Saving

Weights are saved in:

best_model.pkl


Stored parameters:

conv1 weights & bias

conv2 weights & bias

fully connected weights & bias

🛠 Optimization Details

Optimizer: SGD
Learning Rate: configurable
L2 Regularization: implemented via weight_decay
Early Stopping: validation-based

⏱ Performance Constraints

The model is designed to:

Train within 3 hours

Evaluate within 1 hour

Maintain moderate parameter count

Balance accuracy vs computational cost

📈 Observed Performance

Dataset-1 (10 classes):

Validation Accuracy ≈ 94–95%

Dataset-2 (100 classes):

Validation Accuracy ≈ 20–25% (lightweight model)

Higher accuracy achievable with larger models (trade-off with training time)

🔍 Reproducibility

Random seed fixed

Deterministic weight initialization

Dataset split randomized but reproducible

📌 Compliance Statement

This project:

Uses no external ML libraries

Uses only standard C++ and Python libraries

Uses OpenCV strictly for image loading

Implements full backpropagation manually

Complies fully with assignment constraints

📚 Sources Used

Course lectures

Official C++ documentation

pybind11 documentation

Basic CNN architecture references

AI assistance (ChatGPT) for debugging guidance

🏁 Final Notes

This implementation demonstrates:

Full custom autograd engine

Manual CNN implementation

C++–Python integration

Model complexity analysis

Practical training & evaluation pipeline
