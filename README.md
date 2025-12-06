# Parallel Multi-Modal Terrain Roughness Classification

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

<img width="1500" height="938" alt="image" src="https://github.com/user-attachments/assets/5ce1c3c4-06ae-4cc6-a6b3-39d1b7cff444" />


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
  
## High Level Architecture
<img width="1500" height="837" alt="image" src="https://github.com/user-attachments/assets/388b579c-96cb-48ec-986a-c9efd3e3eba9" />

## Results Summary

### Model Performance Comparison

| Model | Features | Accuracy | Macro F1 | Class 0 F1 | Class 1 F1 | Class 2 F1 |
|-------|----------|----------|----------|------------|------------|------------|
| Random Forest (Baseline) | Sensor-only | 63% | 0.58 | 0.72 | 0.52 | 0.49 |
| Random Forest (Class-Weighted) | Sensor-only | 65% | 0.61 | 0.74 | 0.55 | 0.53 |
| Fusion MLP (Baseline) | Sensor + Image | 52% | 0.45 | 0.70 | 0.28 | 0.38 |
| **Fusion MLP (Class-Weighted)** | Sensor + Image | 54% | 0.48 | 0.74 | 0.29 | 0.41 |

### Class-Weighted Fusion MLP Detailed Results
```
              precision    recall  f1-score   support

     Smooth       0.77      0.71      0.74       737
     Medium       0.45      0.22      0.29       492
      Rough       0.27      0.82      0.41       151

   accuracy                           0.54      1380
  macro avg       0.49      0.58      0.48      1380
weighted avg       0.60      0.54      0.54      1380
```

### Confusion Matrix (Fusion MLP - Class Weighted)
```
              Predicted
            Smooth  Medium  Rough
Actual
Smooth        521     118     98
Medium        146     107    239
Rough          12      15    124
```

**Key Observations:**
- Class-weighted models significantly improve recall for minority class (Rough: 82%)
- Fusion MLP captures visual cues that sensor-only models miss
- Class 1 (Medium) remains challenging due to overlap with both Smooth and Rough terrain

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
- **Random Forest**: CPU-parallel with `n_jobs=-1`, `class_weight="balanced"`
- **Fusion MLP**: GPU-accelerated with inverse frequency class weighting

## Model Architectures

### Enhanced Random Forest
```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
```

### Fusion MLP (Class-Weighted)
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

# Class weighting (inverse frequency)
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (len(class_counts) * class_counts)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

## Profiling Results (NSYS)
<img width="1500" height="938" alt="image" src="https://github.com/user-attachments/assets/27fdf444-3594-4c3b-9145-913f4538c01a" />


### NVTX Range Summary (Epoch/Batch Timing)
<img width="975" height="154" alt="image" src="https://github.com/user-attachments/assets/49ea9870-4fce-4f95-9ad6-6b2c5cbfc18a" />


| Range | Time (%) | Total Time (s) | Instances | Avg (s) |
|-------|----------|----------------|-----------|---------|
| Batch | 43.4% | 0.524 | 132 | 0.004 |
| Epoch_0 | 30.2% | 0.365 | 1 | 0.365 |
| Epoch_1 | 13.1% | 0.159 | 1 | 0.159 |
| Epoch_2 | 13.2% | 0.160 | 1 | 0.160 |

> **Note:** Epoch 0 is ~2.3x slower due to CUDA JIT compilation warmup.

### GPU Kernel Distribution

<img width="975" height="571" alt="image" src="https://github.com/user-attachments/assets/9fd1eba0-7900-4fc8-8d50-0cc77228daaa" />


| Kernel | Time (%) | Description |
|--------|----------|-------------|
| `maxwell_sgemm_128x32_tn` | 16.0% | Forward pass matrix multiplication |
| `maxwell_sgemm_128x64_nt` | 13.3% | Backward pass matrix multiplication |
| `multi_tensor_apply_kernel` | ~35% | Adam optimizer updates |
| `sgemm_32x32x32_NN_vec` | 5.8% | Smaller linear layer matmuls |
| `reduce_kernel` | 3.8% | Loss computation reduction |
| `fused_dropout_kernel` | 2.6% | Dropout regularization |

### CUDA API Summary
<img width="975" height="250" alt="image" src="https://github.com/user-attachments/assets/996dbede-1029-4142-8d7f-42dd10c68a06" />


| API Call | Time (%) | Total (s) | Num Calls |
|----------|----------|-----------|-----------|
| `cudaLaunchKernel` | 88.3% | 0.220 | 5,812 |
| `cudaFree` | 6.9% | 0.017 | 3 |
| `cudaMemcpyAsync` | 1.4% | 0.004 | 272 |

### Memory Transfer Efficiency
<img width="975" height="200" alt="image" src="https://github.com/user-attachments/assets/d11602ce-bfab-47ac-b232-c64875d2c994" />


| Operation | Total (MB) | Count | Time (s) |
|-----------|------------|-------|----------|
| Host-to-Device | 40.95 | 272 | 0.004 |

**Optimizations Applied:** `pin_memory=True`, `non_blocking=True`

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

## Future Enhancement
1. Using the FSDP 2 (fullyShardedParallel) for heavier and computational heavy pretrained models
2. Using Automated Mixed Precision for hybrid precision formats (FP16,FP32,T8) for changing value precision and providing auto scaler for different stages such as model training , backpropogation and inference , etc.
3. Using OpenCV, for real time offroad terrain detection during video capture
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
