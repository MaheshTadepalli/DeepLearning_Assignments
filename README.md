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

### 🔹 C++ Backend

* Tensor operations
* Automatic differentiation
* Convolution & pooling
* Activation functions
* Linear layers

### 🔹 Python Frontend

* Dataset loading
* Training & validation
* Evaluation
* Optimization pipeline

---

## ❗ Restrictions Followed

No external deep learning libraries were used:

* PyTorch ❌
* TensorFlow ❌
* NumPy ❌
* SciPy ❌

Only the following were used:

* Standard C++ libraries
* Standard Python libraries
* OpenCV (strictly for image loading)

---

## 🚀 Framework Features

The framework supports:

* Tensor abstraction with gradient tracking
* Automatic differentiation (Backpropagation)
* Convolutional layers
* ReLU activation
* MaxPooling layers
* Fully connected (Linear) layers
* Fused Softmax + Cross Entropy Loss
* SGD optimizer with L2 regularization
* Early stopping
* Model saving and loading
* MACs and FLOPs computation

---

## 🧠 Model Architecture

### Input

`32 × 32 × 3`

---

### Convolution Layer 1

* Kernel size: `3 × 3`
* Channels: `3 → 16`
* Output: `30 × 30 × 16`

ReLU → MaxPool `(2×2, stride 2)`
Output: `15 × 15 × 16`

---

### Convolution Layer 2

* Kernel size: `3 × 3`
* Channels: `16 → 32`
* Output: `13 × 13 × 32`

ReLU → MaxPool → MaxPool

Final Feature Map:
`3 × 3 × 32`

---

### Fully Connected Layer

Flatten → `3 × 3 × 32 → Number of Classes`

Loss: **Fused Softmax Cross Entropy**

---

## 📊 Model Complexity Metrics

Automatically printed during training:

* Total trainable parameters
* MACs (Multiply–Accumulate operations)
* FLOPs per forward pass

---

## 📂 Dataset Format

Dataset must follow:

```
dataset_root/
    class_0/
        img1.png
        img2.png
    class_1/
        img1.png
```

### Requirements

* Images must be PNG format
* Automatically resized to 32×32
* Labels inferred from folder names
* Dataset loading time is measured

---

## ⚙️ Building the Framework

### Step 1 — Build C++ Backend

```bash
cd cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

This generates:

```
mydl_cpp
```

---

### Step 2 — Install Python Dependency

```bash
pip install opencv-python
```

---

## 🏋️ Training

### Steps

1. Open `train.py`
2. Setting Dataset Path

    Go to the **bottom of the file**:
    
    ```python
    if __name__ == "__main__":
    
        train(
            "C:/mine/ASSIGNMENTS/DL/assign1/data_2",  # ← SET YOUR DATASET PATH HERE
            epochs=4,
            batch_size=256,
            lr=0.005,
            weight_decay=2e-4,
            patience=1
        )
    ```
    
    👉 Replace the string with your dataset folder path:
    
    Example:
    
    ```python
    train("D:/dataset/train_data")
    ```
    
    This path must point to the **dataset root folder** containing class subfolders.

4. Run:

```bash
python train.py
```

### Training Output Includes

* Dataset loading time
* Parameter count
* MACs & FLOPs
* Training loss & accuracy
* Validation loss & accuracy
* Epoch time
* Early stopping status

Best model saved as:

```
best_model.pkl
```

---

## 🧪 Evaluation

### Steps

1. Open `evaluation.py`

2. Set:

   * dataset_path
   * weights_path

3. Run:

```bash
python evaluation.py
```

### Evaluation Output

* Dataset loading time
* Parameter count
* MACs & FLOPs
* Evaluation accuracy
* Evaluation time

---

## ⚡ Optimization Details

* Optimizer: **SGD**
* Learning Rate: Configurable
* L2 Regularization: Weight decay
* Early Stopping: Validation-based

---

## 🔁 Reproducibility

* Fixed random seed
* Deterministic initialization
* Explicit train-validation split

---

## ✅ Assignment Compliance

This implementation:

* Uses no external ML frameworks
* Implements manual backpropagation
* Uses OpenCV strictly for image loading
* Computes parameters, MACs, FLOPs
* Supports reproducible training
* Meets assignment time constraints

---

## 📁 Repository Structure

```
cpp/                → C++ backend implementation

python/
    train.py        → Training script
    evaluation.py   → Evaluation script
    model.py        → CNN architecture
    dataset.py      → Dataset loader
    loss.py         → Cross entropy & accuracy
    optimizer.py    → SGD implementation
```

---

## 📝 Notes

The framework is designed to balance:

* Model complexity
* Training time constraints
* Validation accuracy
* Computational efficiency

---

## 🎯 Summary

This project demonstrates a **complete deep learning pipeline** built entirely from first principles without using any deep learning libraries.
