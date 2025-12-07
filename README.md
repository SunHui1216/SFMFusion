<p align="center">

  <h1 align="center">Spatial-Frequency Enhanced Mamba for Multi-Modal Image Fusion</h1>
<p align="center">
    <a href="https://arxiv.org/pdf/2511.06593v1" rel="external nofollow noopener" target="_blank">TIP 2025 Paper</a>

![SFMFusion](Framework.png)

**SFMFusion** is a novel multi-modal image fusion framework designed to integrate complementary information from different modalities. Unlike traditional CNN- or Transformer-based methods that suffer from limited receptive fields or high computational cost, SFMFusion leverages Mamba to model long-range dependencies with linear complexity. Built upon this foundation, SFMFusion enhances Mamba with full spatial and frequency perceptions through the proposed Spatial-Frequency Enhanced Mamba Block, and efficiently couples fusion with image reconstruction via a three-branch structure. In addition, the Dynamic Fusion Mamba Block enables flexible feature aggregation across branches. Extensive experiments on six MMIF datasets demonstrate that SFMFusion achieves superior performance and provides a promising solution for multi-modal image fusion.

## News
Exciting news! Our paper has been accepted by the TIP 2025! 🎉 [Paper](<https://arxiv.org/pdf/2511.06593v1>)

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
  
Please organize the files following the directory structure of the MSRS folder under data.

---

## 🚀 Usage
### 1)Train
python train.py
### 2)Test with pretrained weights
python test.py
### 3)Evaluate metrics
python test_metric.py
