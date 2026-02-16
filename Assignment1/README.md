# GNR638 – Programming Assignment  
## Custom Deep Learning Framework (C++ Backend + Python Frontend)

---

## 📘 Course Information

**Course:** Machine Learning for Remote Sensing – II (Spring 2025–26)  
**Assignment:** Design a Deep Learning Framework  

---

## 📖 Overview

This repository contains a **custom deep learning framework** implemented completely from scratch.

The framework consists of:

- **C++ Backend**
  - Tensor operations
  - Automatic differentiation
  - Convolution & pooling
  - Activation functions
  - Linear layers

- **Python Frontend**
  - Dataset loading
  - Training & validation
  - Evaluation
  - Optimization pipeline

❗ **Restrictions Followed:**

- No external deep learning libraries were used:
  - PyTorch ❌
  - TensorFlow ❌
  - NumPy ❌
  - SciPy ❌
- Only:
  - Standard C++ libraries
  - Standard Python libraries
  - OpenCV (strictly for image loading)

---

## 🚀 Framework Features

The framework supports:

- Tensor abstraction with gradient tracking
- Automatic differentiation (Backpropagation)
- Convolutional layers
- ReLU activation
- MaxPooling layers
- Fully connected (Linear) layers
- Fused Softmax + Cross Entropy Loss
- SGD optimizer with L2 regularization
- Early stopping
- Model saving and loading
- MACs and FLOPs computation

---

## 🧠 Model Architecture

### Input
`32 × 32 × 3`

---

### Convolution Layer 1
- Kernel: `3 × 3`
- Channels: `3 → 16`
- Output: `30 × 30 × 16`

➡ ReLU  
➡ MaxPool `(2 × 2, stride 2)`  
➡ Output: `15 × 15 × 16`

---

### Convolution Layer 2
- Kernel: `3 × 3`
- Channels: `16 → 32`
- Output: `13 × 13 × 32`

➡ ReLU  
➡ MaxPool `(2 × 2)`  
➡ MaxPool `(2 × 2)`

Final Feature Map:  
`3 × 3 × 32`

---

### Fully Connected Layer

Flatten → `3 × 3 × 32 → Number of Classes`

Loss: **Fused Softmax Cross Entropy**

---

## 📊 Model Complexity Metrics

The framework automatically prints:

- Total trainable parameters
- MACs (Multiply–Accumulate operations)
- FLOPs per forward pass

---

## 📂 Dataset Format

Dataset must follow this structure:

dataset_root/
        class_0/
            img1.png
            img2.png
                .
                .
        class_1/
            img1.png
            img2.png
                .
                .


### Requirements

- Images must be **PNG**
- Automatically resized to **32 × 32**
- Labels inferred from folder names
- Dataset loading time is measured

---

## ⚙️ Building the Framework

### Step 1 — Build C++ Backend

cd cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release

This generates the Python module: mydl_cpp

Step 2 — Install Python Dependency
