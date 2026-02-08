# CUDA Vector Addition

A high-performance implementation of vector operations using NVIDIA CUDA.

## Overview
This project demonstrates the core principles of GPU computing:
- *Unified Memory:* Automatically moves data between CPU and GPU memory as needed.
- *Kernel Efficiency:* Implementing optimized thread-to-element mapping.

## Performance
- *Scale:* Processes 1 Million floating-point elements.
- *Architecture:* Optimized for NVIDIA GPUs with Unified Memory support.

## Usage
Compile with NVCC:
nvcc cuda.cu -o vector_benchmark

