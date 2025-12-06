# Parallel Multi-Modal Terrain Roughness Classification

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
<img width="1500" height="938" alt="image" src="https://github.com/user-attachments/assets/0d92e65e-1f14-4c42-b0ff-e95690f189db" />


> **Course:** CSYE 7105 - High Performance Parallel Machine Learning & AI  
> **Team 11:** Sathvik Vadavatha, Shushil Girish  
> **Institution:** Northeastern University  
> **Date:** December 2025

## Overview

This project implements a complete multi-modal machine learning pipeline for terrain roughness classification, combining RGB images with multi-sensor data (accelerometer, gyroscope, magnetometer, GPS). The system classifies terrain into three roughness levels (Smooth, Medium, Rough) for autonomous vehicle navigation applications.

The project emphasizes **High Performance Computing (HPC)** techniques, demonstrating parallelization strategies across CPU and GPU backends with detailed performance analysis.

## Key Features

- **Multi-Modal Fusion**: Combines 512-dimensional ResNet18 image embeddings with 60+ engineered sensor features
- **Parallel Processing**: Implements multiprocessing, joblib, and Dask for CPU parallelization
- **GPU Acceleration**: CUDA-accelerated embedding extraction and MLP training
- **Distributed Training**: DDP implementation with Gloo (CPU) and NCCL (GPU) backends
- **Performance Profiling**: NVIDIA Nsight Systems (nsys) profiling with NVTX markers
- **Comprehensive Benchmarking**: Speedup and efficiency analysis across different parallelization strategies

## Results Summary

| Model | Accuracy | Macro F1 | Best For |
|-------|----------|----------|----------|
| Random Forest (Sensor-only) | 63% | 0.58 | CPU baseline |
| Fusion MLP (Sensor + Image) | 54% | 0.48 | Multi-modal classification |

### Performance Speedups

| Stage | Method | Speedup | 
|-------|--------|---------|
| Sensor Alignment | Multiprocessing (16w) | 6.7x |
| Feature Extraction | Dask (16 batches) | 1.9x |
| Image Embedding | DDP NCCL (4 GPU) | 4.3x |

## Project Structure
```
├── data/
│   ├── Images/                    # RGB terrain images
│   ├── ImageLabels/               # Ground truth labels (tsm1_k3)
│   └── SensorData/                # Multi-sensor CSV files
│       ├── accelerometer_calibrated_split.csv
│       ├── gyroscope_calibrated_split.csv
│       ├── magnetometer_split.csv
│       └── gps.csv
├── notebooks/
│   └── Final_Project_Pipeline.ipynb  # Main pipeline notebook
├── scripts/
│   ├── mlp_profile.py             # Standalone profiling script
│   ├── ddp_embedding_extract.py   # DDP embedding extraction
│   └── nsys_fusion_mlp.sbatch     # SLURM job script for nsys profiling
├── reports/
│   └── CSYE7105_Team11_Final_Project_Report.pdf
├── environment.yaml               # Conda environment specification
└── README.md
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/terrain-roughness-classification.git
cd terrain-roughness-classification
```

### 2. Create Conda Environment
```bash
conda env create -f environment.yaml
conda activate final_project
```

### 3. Download Dataset

Download the [Off-Road Terrain Dataset](https://www.kaggle.com/datasets/magnumresearchgroup/offroad-terrain-dataset-for-autonomous-vehicles/data) and place it in the `data/` directory.

## Usage

### Running the Full Pipeline (Jupyter)
```bash
jupyter lab notebooks/Final_Project_Pipeline.ipynb
```

### Running on HPC Cluster (SLURM)

#### Interactive GPU Session
```bash
salloc -p gpu --gres=gpu:p100:1 --cpus-per-task=8 --mem=32G --time=04:00:00
```

#### Batch Job Submission
```bash
sbatch scripts/nsys_fusion_mlp.sbatch
```

### NVIDIA Nsight Systems Profiling
```bash
# Run profiler
nsys profile -o mlp_report --trace=cuda,nvtx --cuda-memory-usage=true python mlp_profile.py

# View results
nsys stats --force-export=true mlp_report.nsys-rep

# GPU kernel summary
nsys stats --force-export=true --report cuda_gpu_kern_sum mlp_report.nsys-rep

# NVTX markers (Epoch/Batch timing)
nsys stats --force-export=true --report nvtx_pushpop_sum mlp_report.nsys-rep
```

## Pipeline Stages

### 1. Data Loading & Timestamp Alignment
- Parse image timestamps from filenames
- Align sensor data within 1-second windows around each image
- Parallelized with Python `multiprocessing`

### 2. Sensor Feature Extraction
- Statistical features: mean, std, min, max, energy
- Derived features: jerk, magnitude, FFT spectral features
- Parallelized with `joblib` and `Dask`

### 3. Image Embedding Extraction
- Pretrained ResNet18 backbone (frozen)
- 512-dimensional feature vectors
- GPU-accelerated with PyTorch DataLoader
- DDP support for multi-GPU scaling

### 4. Multi-Modal Fusion
- Merge sensor features with image embeddings
- ~575-625 dimensional fused feature vectors
- Selective standardization (sensor features only)

### 5. Model Training
- **Random Forest**: CPU-parallel with `n_jobs=-1`
- **Fusion MLP**: GPU-accelerated with class weighting

## Model Architecture

### Fusion MLP
```python
FusionMLPDeep(
  (net): Sequential(
    (0): Linear(in_features=575, out_features=1024)
    (1): ReLU()
    (2): Dropout(p=0.4)
    (3): Linear(in_features=1024, out_features=512)
    (4): ReLU()
    (5): Dropout(p=0.4)
    (6): Linear(in_features=512, out_features=128)
    (7): ReLU()
    (8): Dropout(p=0.3)
    (9): Linear(in_features=128, out_features=3)
  )
)
```

## Profiling Results (NSYS)

### GPU Kernel Distribution

| Kernel | Time (%) | Description |
|--------|----------|-------------|
| `maxwell_sgemm_*` | ~29% | Matrix multiplications (forward/backward) |
| `multi_tensor_apply_kernel` | ~35% | Adam optimizer updates |
| `fused_dropout_kernel` | ~3% | Dropout regularization |

### Memory Transfer Efficiency

- **Host-to-Device**: 40.95 MB total, 0.004s
- Optimizations: `pin_memory=True`, `non_blocking=True`

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.11 |
| PyTorch | 2.5.1 |
| CUDA | 12.1 |
| Torchvision | 0.16+ |
| scikit-learn | 1.4+ |
| Dask | 2024+ |
| NumPy | 1.26+ |
| Pandas | 2.1+ |

### Hardware (Northeastern Explorer HPC)

- **GPU Nodes**: 4x NVIDIA Tesla P100 (12GB VRAM)
- **CPU Nodes**: 28-32 cores, 128-256GB RAM
- **Scheduler**: SLURM

## References

1. Paszke, A., et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS 2019.
2. Pedregosa, F., et al. "Scikit-learn: Machine Learning in Python." JMLR 2011.
3. [Off-Road Terrain Dataset](https://www.kaggle.com/datasets/magnumresearchgroup/offroad-terrain-dataset-for-autonomous-vehicles)
4. [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
5. [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

## License

This project is for academic purposes as part of CSYE 7105 at Northeastern University.

## Acknowledgments

- Professor Handan Liu (CSYE 7105)
- Northeastern University Research Computing (Explorer HPC)
- Magnum Research Group (Off-Road Terrain Dataset)
