# SFMFusion

Official implementation of **"Spatial-Frequency Enhanced Mamba for Multi-Modal Image Fusion"**

---

## 📌 Introduction
SFMFusion is a Multi-Modal Image Fusion framework based on the Spatial-Frequency enhanced Mamba.  
This repository provides the training and testing code, along with pretrained weights for reproducing the results in our paper.

---

## 🔧 Requirements
- Python 3.9.12
- PyTorch 2.0.1
- CUDA 12.2
- mamba_ssm 2.0.4

---

## 📂 Dataset Preparation
We use the following datasets.
- **MSRS**: [Download here](https://github.com/Linfeng-Tang/MSRS)
- **M3FD**: [Download here](https://github.com/JinyuanLiu-CV/TarDAL)  
- **FMB**: [Download here](https://github.com/JinyuanLiu-CV/SegMiF) 
- **Harvard**: [Download here](https://www.med.harvard.edu/AANLIB/home.html) 

---

## 🚀 Usage
### 1)Train
python train.py
### 2)Test with pretrained weights
python test.py
### 3)Evaluate metrics
python test_metric.py
