/* 
 * Simple Vector Addition (CUDA)
 * Demonstrates basic Unified Memory usage.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Define vector size (1 Million elements)
#define N 1000000
#define THREADS_PER_BLOCK 256

/*
 * CUDA Kernel: Vector Addition with Square Root
 * Computes: a[i] = sqrt(a[i] + b[i])
 * Each thread handles a single index.
 */
__global__ void vectorAddAndSqrt(float *a, float *b, int n)
{
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: prevent accessing memory outside array bounds
    if (i < n)
    {
        a[i] = sqrtf(a[i] + b[i]);
    }
}

int main(void)
{
    float *x, *y;

    // - 1. Memory Allocation 
    // Use Unified Memory (Managed) to share data between CPU and GPU.
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // - 2. Initialization
    // Initialize vectors on CPU.
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // - 3. Kernel Configuration
    // Calculate the number of blocks needed to cover N elements.
    // Uses ceiling division to ensure we have enough threads.
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("%d blocks and %d threads per block...\n", numBlocks, THREADS_PER_BLOCK);

    // - 4. Kernel Launch
    vectorAddAndSqrt<<<numBlocks, THREADS_PER_BLOCK>>>(x, y, N);

    // - 5. Synchronization
    // Wait for the GPU to finish before the CPU accesses the results.
    cudaDeviceSynchronize();

    // - 6. Verification
    // Check the first few results to ensure correctness.
    // Expected result: sqrt(1.0 + 2.0) = sqrt(3.0) approx 1.732
    printf("Verification (First 5 elements):\n");
    for (int i = 0; i < 5; i++)
    {
        printf("Index %d: %.6f\n", i, x[i]);
    }

    // - 7. Cleanup
    cudaFree(x);
    cudaFree(y);

    printf("Done.\n");
    return 0;
}
