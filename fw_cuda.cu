/*

        nvcc -arch compute_70 -code sm_70 fw_cuda.cu -o fw_cuda
        RTX 4060 Ada: nvcc -arch compute_89 -code sm_89 fw_cuda.cu -o fw_cuda

        %s/^M//

        qrsh -l gpus=1 -P ec527

*/

#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

void initializeArray2D(int *arr1, int *arr2, int seed);
void host_FW(int *d, int N);

#define ARR_DIM         1024    // 128 x 128 array
int N                   =       ARR_DIM; // N nodes in the graph
#define ARR_LEN         (ARR_DIM * ARR_DIM)     // 1M elements

#define BLOCK_DIM       32              // 32x32 threads in block
#define THREADS_PER_BLOCK       (BLOCK_DIM * BLOCK_DIM) // 1024 threads in block
#define GRID_DIM        (ARR_DIM + BLOCK_DIM - 1) / BLOCK_DIM   // Enough blocks to cover the array

//#define TOL_ERR_CHECK 0.000001        // fraction of larger element
#define INF 99999999    // Infinity value for the graph

// Access d[i][j] in a flat array
#define IDX(i, j, N)    ((i) * (N) + (j))

/*
        The kernel updates the distance matrix d[i][j] for a fixed k.
*/
__global__ void fw_kernel(int *d, int k, int N) {
        // Each thread computes a single element d[i][j] of the distance matrix

        /* int tx = threadIdx.x;        // Thread index in the block
        int ty = threadIdx.y;   // Thread index in the block
        int bx = blockIdx.x;    // Block index in the grid
        int by = blockIdx.y;    // Block index in the grid

        int i = by * blockDim.y + ty;   // Row index
        int j = bx * blockDim.x + tx;   // Column index */
        int i = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
        int j = blockIdx.x * blockDim.x + threadIdx.x;  // Column index

        if (i < N && j < N) {   // If within bounds of the matrix
                // Compute new distance using the k-th node as an intermediate node
                int dik = d[IDX(i, k, N)];      // d[i][k]
                int dkj = d[IDX(k, j, N)];      // d[k][j]
                int dij = d[IDX(i, j, N)];      // d[i][j]

                // If distance through k is shorter, update distance
                if (dik != INF && dkj != INF && dik + dkj < dij) {
                        dij = dik + dkj;
                }
        }
}

int main (int argc, char **argv) {

        // GPU Timing Variables
        cudaEvent_t startData, endData, startFW, endFW;
        float elapsedGPUData, elapsedGPUFW;

        // Arrays on GPU global memory
        int *d_d;       // Distance matrix on GPU

        // Arrays on host memory
        int *h_d;       // Distance matrix on host
        int *h_d_gold;  // Gold distance matrix for validation

        printf("Length of the array: %d\n", ARR_LEN);

        // Select GPU
        CUDA_SAFE_CALL(cudaSetDevice(0));

        // Allocate GPU memory
        size_t allocSize = ARR_LEN * sizeof(int);
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_d, allocSize));

        // Allocate host memory
        h_d = (int *)malloc(allocSize);
        h_d_gold = (int *)malloc(allocSize);

        // Initialize host arrays
        printf("Initializing host arrays...\n");
        // Arrays are initialized with a known seed for reproducibility
        initializeArray2D(h_d, h_d_gold, 0x12345678);
        printf("... done\n");

        // Create CUDA events for timing
        CUDA_SAFE_CALL(cudaEventCreate(&startData));
        CUDA_SAFE_CALL(cudaEventCreate(&endData));
        CUDA_SAFE_CALL(cudaEventCreate(&startFW));
        CUDA_SAFE_CALL(cudaEventCreate(&endFW));

        // Record start event for data transfer
        CUDA_SAFE_CALL(cudaEventRecord(startData, 0));

        // Transfer arrays to GPU memory
        CUDA_SAFE_CALL(cudaMemcpy(d_d, h_d, allocSize, cudaMemcpyHostToDevice));

        // Record start event for Floyd-Warshall kernel
        CUDA_SAFE_CALL(cudaEventRecord(startFW, 0));

        // Launch kernel for each k from 0 to N-1
        dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);    // Block size
        dim3 dimGrid(GRID_DIM, GRID_DIM);       // Grid size

        for (int k = 0; k < N; k++) {
                // Launch kernel for each k
                fw_kernel<<<dimGrid, dimBlock>>>(d_d, k, N);
                CUDA_SAFE_CALL(cudaGetLastError());     // Check for kernel launch errors
                CUDA_SAFE_CALL(cudaDeviceSynchronize());
        }

        // Record end event for Floyd-Warshall kernel
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        CUDA_SAFE_CALL(cudaEventRecord(endFW, 0));

        // Transfer results back to host
        CUDA_SAFE_CALL(cudaMemcpy(h_d, d_d, allocSize, cudaMemcpyDeviceToHost));

        // Record end event for data transfer
        CUDA_SAFE_CALL(cudaEventRecord(endData, 0));

        // Stop and destroy timers
        CUDA_SAFE_CALL(cudaEventSynchronize(endFW));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedGPUFW, startFW, endFW));
        CUDA_SAFE_CALL(cudaEventSynchronize(endData));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedGPUData, startData, endData));
        printf("\nGPU Time (w/ data transfer): %f ms\n", elapsedGPUData);
        printf("GPU Time (kernel only): %f ms\n", elapsedGPUFW);
        CUDA_SAFE_CALL(cudaEventDestroy(startData));
        CUDA_SAFE_CALL(cudaEventDestroy(endData));
        CUDA_SAFE_CALL(cudaEventDestroy(startFW));
        CUDA_SAFE_CALL(cudaEventDestroy(endFW));


        // Compute results on host for validation
        struct timespec time_start, time_stop;
        clock_gettime(CLOCK_REALTIME, &time_start);

        host_FW(h_d_gold, N);

        clock_gettime(CLOCK_REALTIME, &time_stop);

        double CPU_time_taken;
        {
                struct timespec temp;
                temp.tv_sec = time_stop.tv_sec - time_start.tv_sec;
                temp.tv_nsec = time_stop.tv_nsec - time_start.tv_nsec;
                if (temp.tv_nsec < 0) {
                        temp.tv_sec--;
                        temp.tv_nsec += 1e9;
                }
                CPU_time_taken = temp.tv_sec * 1.0e3 + temp.tv_nsec * 1.0e-6; // sec to ms
        }

        printf("CPU Time: %f ms\n", CPU_time_taken);


        // Validate results
        int errCount = 0;
        int max_diff = 0;

        for (int i = 0; i < ARR_LEN; i++) {
                float diff = abs(h_d[i] - h_d_gold[i]);
                if (diff > 1) {
                        errCount++;
                }
                if (diff > max_diff) {
                        max_diff = diff;
                }
        }

        if (errCount > 0) {
                printf("\nERROR: TEST FAILED: %d results did not match\n", errCount);
        } else {
                printf("\nTEST PASSED: All results fell within tolerance range.\n");
        }
        printf("Maximum difference between CPU and GPU results: %d\n", max_diff);


        // Free device and host memory
        CUDA_SAFE_CALL(cudaFree(d_d));
        free(h_d);
        free(h_d_gold);

        return 0;
}


void initializeArray2D(int *arr1, int *arr2, int seed) {
        srand(seed);

        // Initialize: zero on diagonal, 0.35 chance of INF, otherwise random value between 50 and 99
        for (int i = 0; i < ARR_DIM; i++) {
                //printf("\n");
                for (int j = 0; j < ARR_DIM; j++) {

                        if (i == j) {
                                arr1[IDX(i, j, ARR_DIM)] = 0;
                                arr2[IDX(i, j, ARR_DIM)] = 0;
                        } else {
                                float randNum = (float)rand() / RAND_MAX;       // Random value between 0 and 1
                                if (randNum < 0.35) {
                                        arr1[IDX(i, j, ARR_DIM)] = INF;
                                        arr2[IDX(i, j, ARR_DIM)] = INF;
                                } else {
                                        arr1[IDX(i, j, ARR_DIM)] = rand() % 50 + 50;    // Random value between 50 and 99
                                        arr2[IDX(i, j, ARR_DIM)] = arr1[IDX(i, j, ARR_DIM)];
                                }
                        }

                        //printf("%d ", arr1[IDX(i, j, ARR_DIM)]);
                }
        }
}

void host_FW(int *d, int N) {
        // Host implementation of Floyd-Warshall algorithm
        for (int k = 0; k < N; k++) {
                for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                                if (d[IDX(i, k, N)] != INF && d[IDX(k, j, N)] != INF) {
                                        int dik = d[IDX(i, k, N)];      // d[i][k]
                                        int dkj = d[IDX(k, j, N)];      // d[k][j]
                                        int dij = d[IDX(i, j, N)];      // d[i][j]

                                        if (dik + dkj < dij) {
                                                dij = dik + dkj;        // Update distance
                                        }
                                }
                        }
                }
        }
}