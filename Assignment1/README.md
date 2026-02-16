GNR638 – Programming Assignment
Custom Deep Learning Framework (C++ Backend + Python Frontend)
Course Information

Course: Machine Learning for Remote Sensing – II (Spring 2025–26)
Assignment: Design a Deep Learning Framework

Overview

This repository contains a custom deep learning framework implemented completely from scratch.

The framework consists of:

A C++ backend implementing tensor operations, automatic differentiation, convolution, pooling, activation, and linear layers.

A Python frontend handling dataset loading, training, validation, evaluation, and optimization.

No external deep learning libraries (PyTorch, TensorFlow, NumPy, SciPy, etc.) were used.
Only standard C++/Python libraries and OpenCV (strictly for image loading) are used, in accordance with assignment constraints.

Framework Features

The framework supports:

Tensor abstraction with gradient tracking

Automatic differentiation (backpropagation)

Convolutional layers

ReLU activation

MaxPooling layers

Fully connected (linear) layers

Fused Softmax + Cross Entropy loss

SGD optimizer with L2 regularization

Early stopping

Model saving and loading

MACs and FLOPs computation

Model Architecture

The CNN architecture used is:

Input: 32 × 32 × 3

Convolution Layer 1

Kernel size: 3 × 3

Channels: 3 → 16

Output: 30 × 30 × 16

ReLU

MaxPool (2 × 2, stride 2)

Output: 15 × 15 × 16

Convolution Layer 2

Kernel size: 3 × 3

Channels: 16 → 32

Output: 13 × 13 × 32

ReLU

MaxPool (2 × 2, stride 2)

MaxPool (2 × 2, stride 2)

Final feature map: 3 × 3 × 32

Flatten

Fully Connected Layer

3 × 3 × 32 → Number of classes

Loss is computed using fused Softmax Cross Entropy.

Model Complexity

The following metrics are automatically printed during training and evaluation:

Total number of trainable parameters

MACs (Multiply–Accumulate operations)

FLOPs per forward pass

These values depend on the number of output classes.

Dataset Format

The dataset must follow an image-folder structure:

dataset_root/
    class_0/
        img1.png
        img2.png
    class_1/
        ...


Requirements:

Images must be PNG format

Automatically resized to 32 × 32

Labels are inferred from folder names

Dataset loading time is measured and printed

Building the Framework
Step 1: Build C++ Backend

From the cpp directory:

mkdir build
cd build
cmake ..
cmake --build . --config Release


This generates the Python module:

mydl_cpp

Step 2: Install Python Dependency

Only OpenCV is required:

pip install opencv-python

Training

To train the model:

Open train.py

Set the dataset path

Run:

python train.py


The training script prints:

Dataset loading time

Parameter count

MACs and FLOPs

Training loss

Training accuracy

Validation loss

Validation accuracy

Epoch time

Early stopping status

The best performing model (based on validation accuracy) is saved as:

best_model.pkl

Evaluation

To evaluate on a test dataset:

Open evaluation.py

Set:

dataset_path

weights_path

Run:

python evaluation.py


The script prints:

Dataset loading time

Parameter count

MACs and FLOPs

Evaluation accuracy

Evaluation time

No code modification is required by the grader beyond dataset and weight paths.

Optimization Details

Optimizer: Stochastic Gradient Descent (SGD)

Learning Rate: configurable

L2 Regularization: implemented via weight decay

Early Stopping: validation-based

Reproducibility

Fixed random seed

Deterministic initialization

Explicit training-validation split

Assignment Compliance

This implementation:

Uses no external ML frameworks

Implements backpropagation manually

Uses OpenCV strictly for image loading

Computes model complexity (Parameters, MACs, FLOPs)

Supports reproducible training and evaluation

Meets assignment time constraints

Repository Structure
cpp/                → C++ backend implementation
python/
    train.py        → Training script
    evaluation.py   → Evaluation script
    model.py        → CNN architecture
    dataset.py      → Dataset loader
    loss.py         → Cross entropy & accuracy
    optimizer.py    → SGD implementation

Notes

The framework is designed to balance:

Model complexity

Training time constraints

Validation accuracy

Computational efficiency

This implementation demonstrates a full deep learning pipeline built entirely from first principles.
