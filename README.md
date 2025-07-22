# LRDNet: Lightweight LiDAR Aided Cascaded Feature Pools for Free Road Space Detection

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TMM%202025-blue.svg)](https://ieeexplore.ieee.org/document/)
[![Framework](https://img.shields.io/badge/Framework-Keras-red.svg)](https://keras.io/)
[![Dataset](https://img.shields.io/badge/Dataset-KITTI-green.svg)](http://www.cvlibs.net/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Abstract

This repository presents the official implementation of **LRDNet**, a novel lightweight deep learning architecture specifically engineered for efficient free road space detection in autonomous driving scenarios. Our methodology addresses the critical computational constraints inherent in real-time embedded deployment while maintaining competitive segmentation accuracy across established benchmarks.

The proposed network architecture demonstrates remarkable parameter efficiency, utilizing merely **19.5M parameters** while achieving state-of-the-art processing speeds suitable for resource-constrained environments. Through innovative cascaded feature pooling mechanisms and strategic architectural design choices, LRDNet establishes new performance benchmarks in the intersection of computational efficiency and semantic understanding for road scene analysis.

## Key Contributions

- **Parameter-Efficient Architecture**: Revolutionary lightweight design achieving competitive performance with minimal computational overhead
- **Embedded System Optimization**: Framework selection and architectural decisions specifically tailored for deployment on resource-constrained hardware platforms
- **Real-Time Processing Capability**: Demonstrates exceptional inference speeds reaching up to 300 FPS on optimized hardware configurations
- **Comprehensive Benchmark Evaluation**: Extensive validation across multiple standard datasets including KITTI, Cityscapes, and R2D benchmarks

## Experimental Results

### Qualitative Performance Analysis
![Sample Results](https://github.com/abdkhanstd/LRDNet/raw/main/images/qres.jpg)
*Representative segmentation outputs demonstrating LRDNet's capability across diverse road scenarios on KITTI benchmark*

### Quantitative Performance Metrics
![Performance Comparison](https://github.com/abdkhanstd/LRDNet/raw/main/images/table.png)
*Comprehensive performance analysis comparing parameter count, processing speed, and accuracy metrics against state-of-the-art methods*

**Experimental Configuration**: Evaluation conducted on NVIDIA GeForce RTX 2080Ti with 188GB system memory, utilizing 48-core Intel Xeon processor architecture.

## Technical Implementation

### Repository Architecture
```
LRDNet/
├── train.py                 # Primary training pipeline with configurable hyperparameters
├── trainc.py               # Continuation training from pre-trained checkpoints
├── test.py                 # Inference and evaluation framework
├── AUG/                    # Static data augmentation generation utilities (MATLAB)
├── ADI/                    # Modified Adaptive Data Integration implementation
├── data/                   # Dataset organization directory
│   ├── training/           # Training dataset placement
│   ├── testing/            # Evaluation dataset storage
│   └── data_road_aug/      # Augmented dataset repository
│       ├── train/          # Augmented training samples
│       └── val/            # Augmented validation samples
└── seg_results_images/     # Generated segmentation outputs
```

### Framework Dependencies

**Core Requirements**:
```bash
tensorflow-gpu==1.14.0
keras==2.2.4
tqdm
pillow
numpy
```

**Optional Performance Analysis Tools**:

*FLOPS Computation*: Download `net_flops.py` from the [Keras FLOPS repository](https://github.com/ckyrkou/Keras) and position within root directory for computational complexity analysis.

*Backbone Network Integration*: Install segmentation models library following documentation from [qubvel/segmentation_models](https://github.com/qubvel/segmentation_models).

*Advanced Data Augmentation*: Implement Albumentations library as detailed in [official documentation](https://albumentations.ai/) for sophisticated augmentation strategies.

## Dataset Configuration

### Supported Benchmark Datasets

**Primary Evaluation Datasets**:
- **[KITTI Road Benchmark](http://www.cvlibs.net/)**: Comprehensive autonomous driving dataset for road segmentation evaluation
- **[Cityscapes Dataset](https://www.cityscapes-dataset.com/)**: Large-scale urban scene understanding benchmark
- **[R2D Dataset](https://sites.google.com/view/sne-roadseg/dataset)**: Specialized road segmentation evaluation framework

### Data Organization Protocol

![Directory Structure](https://github.com/abdkhanstd/LRDNet/raw/main/images/folder.png)

Establish dataset organization following the prescribed directory hierarchy to ensure compatibility with training and evaluation pipelines.

## Experimental Protocol

### Model Training Configuration
```bash
python train.py
```
*Configure model hyperparameters and architectural variants within the designated model variable section*

### Checkpoint-Based Training Continuation
```bash
python trainc.py
```
*Specify pre-trained weight file paths for continued optimization from existing checkpoints*

### Inference and Evaluation Pipeline
```bash
python test.py
```
*Define model path specifications for trained network evaluation on test datasets*

## Evaluation Methodology

### Birds Eye View (BEV) Transformation Protocol

Generated segmentation outputs are systematically stored within the `seg_results_images/` directory. For compatibility with KITTI evaluation protocols, implement BEV coordinate transformation following guidelines established in the [KITTI Road Development Kit](http://www.cvlibs.net/).

### Benchmark Submission Framework

1. Establish official evaluation account using institutional email credentials
2. Implement evaluation protocols as specified in [KITTI Road Kit documentation](http://www.cvlibs.net/)
3. Submit BEV-transformed segmentation results for official benchmark evaluation

## Pre-Trained Model Variants

We provide comprehensive pre-trained network weights representing different architectural configurations evaluated on the KITTI benchmark server:

- **LRDNet+**: Enhanced architectural variant incorporating advanced feature extraction mechanisms
- **LRDNet(S)**: Standard configuration optimized for balanced performance and efficiency
- **LRDNet(L)**: Large-scale variant designed for maximum accuracy scenarios

**Resource Access**: Complete model weights, BEV submission files, modified ADI implementations, and KITTI submission documentation available through [institutional cloud storage](https://stduestceducn-my.sharepoint.com/:f:/g/personal/201714060114_std_uestc_edu_cn/EhqB09h_M_hKistKRBZd-VwB1J3mDkXTy-TwoML1ZR8_tA?e=WGX03e).

## Technical Implementation Notes

**Framework Selection Rationale**: While acknowledging PyTorch's superior computational performance in traditional GPU environments, our implementation utilizes Keras/TensorFlow for enhanced compatibility with embedded deployment scenarios. This architectural decision facilitates seamless integration with resource-constrained hardware platforms capable of achieving the target 300 FPS processing requirements.

**Code Verification Status**: The current repository contains cleaned implementation code that has not undergone comprehensive testing post-refactoring. Users encountering implementation issues are encouraged to report problems through the repository's issue tracking system.

## Citation

Please reference our work using the following citation format:

```bibtex
@article{DBLP:journals/tmm/KhanSRSS25,
  author       = {Abdullah Aman Khan and
                  Jie Shao and
                  Yunbo Rao and
                  Lei She and
                  Heng Tao Shen},
  title        = {LRDNet: Lightweight LiDAR Aided Cascaded Feature Pools for Free Road
                  Space Detection},
  journal      = {{IEEE} Trans. Multim.},
  volume       = {27},
  pages        = {652--664},
  year         = {2025},
}
```

## Contact & Support

For technical inquiries, implementation questions, or collaboration opportunities, please utilize the repository's issue tracking system or contact the corresponding author through institutional channels.

---

**Disclaimer**: This implementation represents ongoing research in autonomous driving perception systems. Users are advised to conduct thorough validation before deployment in safety-critical applications.
