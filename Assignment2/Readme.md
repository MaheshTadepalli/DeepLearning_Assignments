Pre-trained CNN Representation Transfer and Robustness Analysis

This repository contains the implementation for Coding Assignment-2 (GNR638). The project studies how different pre-trained convolutional neural networks (CNNs) transfer learned representations to a new image classification task.

The experiments analyze transfer learning performance, fine-tuning strategies, few-shot learning behavior, robustness to corrupted inputs, and layer-wise feature representations.

Models Used

The following ImageNet-pretrained models were used:

ResNet50

DenseNet121

EfficientNet-B0

Models are loaded using the timm library.

Dataset

The experiments use the Aerial Image Dataset (AID).

Dataset characteristics:

30 scene classes

Image classification task

Folder structure format

Example:

AID/
 ├── airport/
 ├── bridge/
 ├── forest/
 ├── farmland/
 └── ...

The dataset is split into:

Training set

Validation set

Hidden test set (used for grading)

Experimental Scenarios

Five experimental scenarios were implemented for each model.

1. Linear Probe Transfer

Backbone frozen

Train only the linear classifier

Evaluate feature transferability

Outputs:

Accuracy vs epoch

Confusion matrix

PCA feature visualization

2. Fine-Tuning Strategies

Four strategies were compared:

Linear probe

Last block fine-tuning

Selective 20% parameter unfreeze

Full fine-tuning

Outputs:

Accuracy vs trainable parameters

Training loss curves

Gradient statistics

3. Few-Shot Learning

Models were trained with different dataset sizes:

100% data

20% data

5% data

Performance drop was calculated to analyze data efficiency.

4. Corruption Robustness

Model robustness was evaluated using corrupted validation images:

Gaussian noise (σ = 0.05, 0.1, 0.2)

Motion blur

Brightness shift

Metrics:

Corruption Error

Relative Robustness

5. Layer-Wise Feature Probing

Feature representations were extracted from:

Early layer

Middle layer

Final layer

A linear classifier was trained on each feature representation.

Outputs:

Accuracy vs layer depth

Feature norm statistics

PCA visualization

Repository Structure
├── ResNet50.ipynb
├── DenseNet121.ipynb
├── EfficientNet.ipynb
└── README.md
Requirements

Python 3.12

Main libraries used:

torch
torchvision
timm
numpy
matplotlib
scikit-learn
thop

Install dependencies:

pip install torch torchvision timm scikit-learn matplotlib thop
Reproducibility

All experiments use fixed random seeds to ensure reproducible results.

Hardware

Experiments were executed on GPU (Kaggle Notebook) using NVIDIA GPUs.
