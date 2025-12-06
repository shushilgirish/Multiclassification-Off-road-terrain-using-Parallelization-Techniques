#!/bin/bash
#SBATCH --job-name=nsys_profile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=nsys_profile_%j.out
#SBATCH --error=nsys_profile_%j.err

# Load modules
module load anaconda3/2024.06 cuda
source activate final_project

# GPU kernel summary
nsys stats --force-export=true --report cuda_gpu_kern_sum mlp_report.nsys-rep

# Your NVTX markers (Epoch/Batch)
nsys stats --force-export=true --report nvtx_pushpop_sum mlp_report.nsys-rep