/* ADOPTED SERIAL CODE IS AS OF NOON 4/22/25 (HILLEL)

   nvcc -arch compute_70 -code sm_70 fw_cuda_tests.cu -o fw_cuda_tests

   qrsh -l gpus=1 -P ec527

EC527 Final Project
Scalar optimizations for Floyd-Warshall Algorithm
- Finds the All-Pairs-Shortest-Paths (APSP) of a randomly generated directed, weighted graph of multiple sizes (represented by adjacency matrices)
- Records the number of cycles taken to run the algorithm for each size

Time measurement code is borrowed from previous EC527 labs.

--------------------------------------------------------------------------------
gcc -O1 fw_scalar_optimizations.c -lrt -o fw_scalar_optimizations

*/

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <time.h>
#include <math.h>

/* =============== Serial Constants =============== */
#define A  32  /* coefficient of x^2 */
#define B  32  /* coefficient of x */
#define C  32  /* constant term */

#define NUM_TESTS 8 // set to 15

#define CPNS 3.0

#define OPTIONS 7

#define IDENT 0

#define INF_EDGE (10 * num_vertices) // arbitrarily large value for infinity edges

#define BLOCK_SIZE 64

typedef int data_t;

/* ============== CUDA Constants =============== */
#define BLOCK_DIM  32   // 32x32 threads per block

#define IDX(i, j, N)    ((i) * (N) + (j))

#define GPU_OPTIONS 1

/* =================== CUDA Function Prototypes =================== */
void flatten_matrix(int M, int N, int 2d[M][N], int *flat);
void host_FW(int *d, int N);
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

/* == Basic Kernel: Updates the distance matrix d[i][j] for a fixed k. == */
__global__ void fw_kernel_basic(int *d, int k, int N) {
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
                        d[IDX(i, j, N)] = dik + dkj;
                }
        }
}


/* =================== Serial Function Prototypes =================== */
int clock_gettime(clockid_t clk_id, struct timespec *tp);
int **create_adjacency_matrix(int num_vertices);
void free_adjacency_matrix(int **matrix, int num_vertices);
void print_graph(int **graph, int num_vertices);
void fw_serial(int **graph, int num_vertices);
void fw_local_variables(int **graph, int num_vertices);
void fw_conditional_move(int **graph, int num_vertices);
void fw_loop_unroll2(int **graph, int num_vertices);
void fw_loop_unroll4(int **graph, int num_vertices);
void fw_loop_unroll8(int **graph, int num_vertices);
void fw_loop_unroll4_lvars(int **graph, int num_vertices);
void fw_blocked(int **graph, int num_vertices);
void process_block(int **graph, int num_vertices, int i, int j, int k);
void process_block_lvars(int **graph, int num_vertices, int i, int j, int k);
void process_block_unroll4(int **graph, int num_vertices, int i, int j, int k);

/* =================== Serial Time Measurement =================== */

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

/* =================== Serial Wakeup Delay =================== */

double wakeup_delay()
{
  double meas = 0; int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
  j = 100;
  while (meas < 1.0) {
    for (i = 1; i < j; i++) {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random*quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    meas = interval(time_start, time_stop);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}

/* =================== Main Function =================== */
void fw_CPU() {
    int OPTION;
    int num_vertices, max_vertices;
    struct timespec time_start, time_stop;
    double time_stamp[OPTIONS][NUM_TESTS];
    double wd;
    int **graph;
    int **ref_graph[NUM_TESTS];
    int x;

    printf("Floyd-Warshall Algorithm - Serial Implementation\n");

    wd = wakeup_delay();

    x = NUM_TESTS - 1;
    max_vertices = A*x*x + B*x + C;

    for (OPTION = 0; OPTION < OPTIONS; OPTION++) {
        printf("Testing option %d\n", OPTION);
        for (x = 0; x < NUM_TESTS && (num_vertices = A*x*x + B*x + C, num_vertices <= max_vertices); x++) {

            // create the adjacency matrix
            graph = create_adjacency_matrix(num_vertices);

            // start timing the algorithm
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);

            switch(OPTION) {
                case 0: // serial/baseline implementation
                    fw_serial(graph, num_vertices);
                    break;
                case 1: // using local variables
                    fw_local_variables(graph, num_vertices); // local variables implementation
                    break;
                case 2: // unrolled implementation by factor of 2
                    fw_loop_unroll2(graph, num_vertices);
                    break;
                case 3: // unrolled implementation by factor of 4
                    fw_loop_unroll4(graph, num_vertices);
                    break;
                case 4: // unrolled implementation by factor of 8
                    fw_loop_unroll8(graph, num_vertices);
                    break;
                case 5: // unrolled implementation by factor of 4 with local variables
                    fw_loop_unroll4_lvars(graph, num_vertices);
                    break;
                case 6: // blocked implementation
                    fw_blocked(graph, num_vertices);
                    break;
                default:
                    break;
            }

            // stop timing the algorithm
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);

            // calculate and store the time taken
            time_stamp[OPTION][x] = interval(time_start, time_stop);

            // copy resulting graph to reference graph for comparison
            if (OPTION == 0) {
                ref_graph[x] = (int **)malloc(num_vertices * sizeof(int *));
                for (int i = 0; i < num_vertices; i++) {
                    ref_graph[x][i] = (int *)malloc(num_vertices * sizeof(int));
                    for (int j = 0; j < num_vertices; j++) {
                        ref_graph[x][i][j] = graph[i][j];
                    }
                }
            }

            // check if the results are correct (based on results from serial implementation)
            if (OPTION != 0) {
                for (int i = 0; i < num_vertices; i++) {
                    for (int j = 0; j < num_vertices; j++) {
                        if (graph[i][j] != ref_graph[x][i][j]) {
                            printf("Error: Results do not match for option %d at (%d, %d)\n", OPTION, i, j);
                            time_stamp[OPTION][x] = 0;
                            break;
                        }
                    }
                }
            }

            printf("  iter %d done\r", x); fflush(stdout);

            // Free the adjacency matrix memory
            free_adjacency_matrix(graph, num_vertices);
        }
    }

    printf("\nnum_vertices, baseline, local variables, unroll 2x, unroll 4x, unroll 8x, unroll 4x with local vars, blocked \n");
    for (x = 0; x < NUM_TESTS && (num_vertices = A*x*x + B*x + C, num_vertices <= max_vertices); x++) {
        printf("%d", num_vertices);
        for (OPTION = 0; OPTION < OPTIONS; OPTION++) {
            printf(", %ld", (long int)((double)(CPNS) * 1.0e9 * time_stamp[OPTION][x]));
        }
        printf("\n");
    }

    printf("\n");
    printf("Initial delay was calculating: %g \n", wd);

}

void fw_GPU() {
    int OPTION;
    int num_vertices, max_vertices;
    int **graph;
    float time_stamp_GPU_data[OPTIONS][NUM_TESTS];
    float time_stamp_GPU_calc[OPTIONS][NUM_TESTS];
    int x;

    // GPU Timing Variables
    cudaEvent_t startData, endData, startFW, endFW;
    float elapsedGPUData, elapsedGPUFW;

    // Arrays on GPU global memory
    int *d_d;       // Distance matrix on GPU
    // Arrays on host memory
    int *h_d;       // Distance matrix on host
    int *h_d_gold;  // Gold distance matrix for verification

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

    printf("Floyd-Warshall Algorithm - GPU Implementation\n");

    x = NUM_TESTS - 1;
    max_vertices = A*x*x + B*x + C;

    for (OPTION = 0; OPTION < GPU_OPTIONS; OPTION++) {
        printf("Testing GPU option %d\n", OPTION);
        for (x = 0; x < NUM_TESTS && (num_vertices = A*x*x + B*x + C, num_vertices <= max_vertices); x++) {
            int N = x;

            // Allocate GPU memory
            size_t allocSize = N*N * sizeof(int);
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_d, allocSize));
            // Allocate host memory
            h_d = (int *)malloc(allocSize);
            h_d_gold = (int *)malloc(allocSize);

            // create the adjacency matrix
            graph = create_adjacency_matrix(num_vertices);
            // Initialize host arrays
            flatten_matrix(x, x, graph, h_d);
            flatten_matrix(x, x, graph, h_d_gold);

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

            switch(OPTION) {
                case 0: // basic GPU implementation
                    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
                    dim3 dimGrid( (N + BLOCK_DIM - 1) / BLOCK_DIM,
                                  (N + BLOCK_DIM - 1) / BLOCK_DIM );
                    for (int k = 0; k < N; k++) {
                        // Launch kernel for each k iteration
                        fw_kernel_basic<<<dimGrid, dimBlock>>>(d_d, k, N);
                        CUDA_SAFE_CALL(cudaGetLastError());
                        CUDA_SAFE_CALL(cudaDeviceSynchronize());
                    }
                    break;
                default:
                    break;
            }

            // Record end event for Floyd-Warshall kernel
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            CUDA_SAFE_CALL(cudaEventRecord(endFW, 0));

            // Transfer results back to host
            CUDA_SAFE_CALL(cudaMemcpy(h_d, d_d, allocSize, cudaMemcpyDeviceToHost));

            // Record end event for data transfer
            CUDA_SAFE_CALL(cudaEventRecord(endData, 0));

            // Calculate and store time taken
            time_stamp_GPU_data[OPTION][x] = elapsedGPUData;
            time_stamp_GPU_calc[OPTION][x] = elapsedGPUFW;

            // Stop and destroy timers
            CUDA_SAFE_CALL(cudaEventSynchronize(endFW));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedGPUFW, startFW, endFW));
            CUDA_SAFE_CALL(cudaEventSynchronize(endData));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedGPUData, startData, endData));
            //printf("\nGPU Time (w/ data transfer): %f ms\n", elapsedGPUData);
            //printf("GPU Time (kernel only): %f ms\n", elapsedGPUFW);
            CUDA_SAFE_CALL(cudaEventDestroy(startData));
            CUDA_SAFE_CALL(cudaEventDestroy(endData));
            CUDA_SAFE_CALL(cudaEventDestroy(startFW));
            CUDA_SAFE_CALL(cudaEventDestroy(endFW));

            // Verify GPU results
            host_FW(h_d_gold, N);
            int errCount = 0;
            int max_diff = 0;
            for (int i = 0; i < N; i++) {
                float diff = abs(h_d[i] - h_d_gold[i]);
                if (diff > 1) errCount++;
                if (diff > max_diff) max_diff = diff;
            }
            if (errCount > 0) {
                printf("\nERROR: %d elements do not match\n", errCount);
            } else {
                printf("\nTEST PASSED: All elements match\n");
            }
            printf("Max difference between CPU and GPU results: %d\n", max_diff);

            // Free device and host memory
            CUDA_SAFE_CALL(cudaFree(d_d));
            free(h_d);
            free(h_d_gold);


            printf("  iter %d done\r", x); fflush(stdout);

            // Free the adjacency matrix memory
            free_adjacency_matrix(graph, num_vertices);
        }
    }

    printf("\nGPU Time (ms):\nnum_vertices, GPU basic\n");
    for (x = 0; x < NUM_TESTS && (num_vertices = A*x*x + B*x + C, num_vertices <= max_vertices); x++) {
        printf("%d", num_vertices);
        for (OPTION = 0; OPTION < OPTIONS; OPTION++) {
            printf(", %f/%f", time_stamp_GPU_data[OPTION][x], time_stamp_GPU_calc[OPTION][x]);
        }
        printf("\n");
    }
}

int main() {

    fw_CPU();
    fw_GPU();

    return 0;
}


/* =================== CUDA Function Definitions =================== */
void flatten_matrix(int M, int N, int 2d[M][N], int *flat) {
        int i, j;
        for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                        flat[i * N + j] = 2d[i][j];
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
                                            d[IDX(i, j, N)] = dik + dkj;        // Update distance
                                    }
                            }
                    }
            }
    }
}

/* =================== Serial Function Definitions =================== */
void fw_serial(int **graph, int num_vertices) {
    int i, j, k;
    for (k = 0; k < num_vertices; k++) {
        for (i = 0; i < num_vertices; i++) {
            for (j = 0; j < num_vertices; j++) {
                if (graph[i][k] + graph[k][j] < graph[i][j]) {
                    graph[i][j] = graph[i][k] + graph[k][j];
                }
            }
        }
    }
}

void fw_conditional_move(int **graph, int num_vertices) {
    int i, j, k, sum;
    for (k = 0; k < num_vertices; k++) {
        for (i = 0; i < num_vertices; i++) {
            for (j = 0; j < num_vertices; j++) {
                sum = graph[i][k] + graph[k][j];
                // user ternary operator to conditionally move the value
                graph[i][j] = (sum < graph[i][j]) ? sum : graph[i][j];
            }
        }
    }
}

void fw_local_variables(int **graph, int num_vertices) {
    int i, j, k;
    for (k = 0; k < num_vertices; k++) {
        for (i = 0; i < num_vertices; i++) {
            int ik = graph[i][k];
            for (j = 0; j < num_vertices; j++) {
                int sum = ik + graph[k][j];
                if (sum < graph[i][j]) {
                    graph[i][j] = sum;
                }
            }
        }
    }
}

void fw_loop_unroll2(int **graph, int num_vertices) {
    int i, j, k;
    // unroll the innermost loop by a factor of 2
    for (k = 0; k < num_vertices; k++) {
        for (i = 0; i < num_vertices; i++) {
            for (j = 0; j < num_vertices; j += 2) {
                if (graph[i][k] + graph[k][j] < graph[i][j]) {
                    graph[i][j] = graph[i][k] + graph[k][j];
                }
                if (j + 1 < num_vertices && graph[i][k] + graph[k][j + 1] < graph[i][j + 1]) {
                    graph[i][j + 1] = graph[i][k] + graph[k][j + 1];
                }
            }
        }
    }
}

void fw_loop_unroll4(int **graph, int num_vertices) {
    int i, j, k;
    // unroll the innermost loop by a factor of 4
    for (k = 0; k < num_vertices; k++) {
        for (i = 0; i < num_vertices; i++) {
            for (j = 0; j < num_vertices; j += 4) {
                if (graph[i][k] + graph[k][j] < graph[i][j]) {
                    graph[i][j] = graph[i][k] + graph[k][j];
                }
                if (j + 1 < num_vertices && graph[i][k] + graph[k][j + 1] < graph[i][j + 1]) {
                    graph[i][j + 1] = graph[i][k] + graph[k][j + 1];
                }
                if (j + 2 < num_vertices && graph[i][k] + graph[k][j + 2] < graph[i][j + 2]) {
                    graph[i][j + 2] = graph[i][k] + graph[k][j + 2];
                }
                if (j + 3 < num_vertices && graph[i][k] + graph[k][j + 3] < graph[i][j + 3]) {
                    graph[i][j + 3] = graph[i][k] + graph[k][j + 3];
                }
            }
        }
    }
}

void fw_loop_unroll8(int **graph, int num_vertices) {
    int i, j, k;
    // unroll the innermost loop by a factor of 8
    for (k = 0; k < num_vertices; k++) {
        for (i = 0; i < num_vertices; i++) {
            for (j = 0; j < num_vertices; j += 8) {
                if (graph[i][k] + graph[k][j] < graph[i][j]) {
                    graph[i][j] = graph[i][k] + graph[k][j];
                }
                if (j + 1 < num_vertices && graph[i][k] + graph[k][j + 1] < graph[i][j + 1]) {
                    graph[i][j + 1] = graph[i][k] + graph[k][j + 1];
                }
                if (j + 2 < num_vertices && graph[i][k] + graph[k][j + 2] < graph[i][j + 2]) {
                    graph[i][j + 2] = graph[i][k] + graph[k][j + 2];
                }
                if (j + 3 < num_vertices && graph[i][k] + graph[k][j + 3] < graph[i][j + 3]) {
                    graph[i][j + 3] = graph[i][k] + graph[k][j + 3];
                }
                if (j + 4 < num_vertices && graph[i][k] + graph[k][j + 4] < graph[i][j + 4]) {
                    graph[i][j + 4] = graph[i][k] + graph[k][j + 4];
                }
                if (j + 5 < num_vertices && graph[i][k] + graph[k][j + 5] < graph[i][j + 5]) {
                    graph[i][j + 5] = graph[i][k] + graph[k][j + 5];
                }
                if (j + 6 < num_vertices && graph[i][k] + graph[k][j + 6] < graph[i][j + 6]) {
                    graph[i][j + 6] = graph[i][k] + graph[k][j + 6];
                }
                if (j + 7 < num_vertices && graph[i][k] + graph[k][j + 7] < graph[i][j + 7]) {
                    graph[i][j + 7] = graph[i][k] + graph[k][j + 7];
                }
            }
        }
    }
}

void fw_loop_unroll4_lvars(int **graph, int num_vertices) {
    int i, j, k;
    // unroll the innermost loop by a factor of 4 with local variables
    for (k = 0; k < num_vertices; k++) {
        for (i = 0; i < num_vertices; i++) {
            int ik = graph[i][k];
            for (j = 0; j < num_vertices; j += 4) {
                int sum1 = ik + graph[k][j];
                int sum2 = ik + graph[k][j + 1];
                int sum3 = ik + graph[k][j + 2];
                int sum4 = ik + graph[k][j + 3];
                if (sum1 < graph[i][j]) {
                    graph[i][j] = sum1;
                }
                if (j + 1 < num_vertices && sum2 < graph[i][j + 1]) {
                    graph[i][j + 1] = sum2;
                }
                if (j + 2 < num_vertices && sum3 < graph[i][j + 2]) {
                    graph[i][j + 2] = sum3;
                }
                if (j + 3 < num_vertices && sum4 < graph[i][j + 3]) {
                    graph[i][j + 3] = sum4;
                }
            }
        }
    }
}

void fw_blocked(int **graph, int num_vertices) {
    int i, j, k;
    for (k = 0; k < num_vertices; k += BLOCK_SIZE) {
        // process the diagonal block
        process_block_lvars(graph, num_vertices, k, k, k);

        // process row and column blocks
        for (i = 0; i < num_vertices; i += BLOCK_SIZE) {
            if (i != k) {
                process_block_lvars(graph, num_vertices, i, k, k);
                process_block_lvars(graph, num_vertices, k, i, k);
            }
        }

        // process the remaining blocks
        for (i = 0; i < num_vertices; i += BLOCK_SIZE) {
            if (i == k) continue;
            for (j = 0; j < num_vertices; j += BLOCK_SIZE) {
                if (j == k) continue;
                process_block_lvars(graph, num_vertices, i, j, k);
            }
        }
    }
}

void process_block(int **graph, int num_vertices, int i, int j, int k) {
    for (int kk = k; kk < k + BLOCK_SIZE && kk < num_vertices; kk++) {
        for (int ii = i; ii < i + BLOCK_SIZE && ii < num_vertices; ii++) {
            for (int jj = j; jj < j + BLOCK_SIZE && jj < num_vertices; jj++) {
                if (graph[ii][kk] + graph[kk][jj] < graph[ii][jj]) {
                    graph[ii][jj] = graph[ii][kk] + graph[kk][jj];
                }
            }
        }
    }
}

void process_block_lvars(int **graph, int num_vertices, int i, int j, int k) {
    for (int kk = k; kk < k + BLOCK_SIZE && kk < num_vertices; kk++) {
        for (int ii = i; ii < i + BLOCK_SIZE && ii < num_vertices; ii++) {
            int ik = graph[ii][kk];
            for (int jj = j; jj < j + BLOCK_SIZE && jj < num_vertices; jj++) {
                int sum = ik + graph[kk][jj];
                if (sum < graph[ii][jj]) {
                    graph[ii][jj] = sum;
                }
            }
        }
    }
}

// Create an adjacency matrix for the graph
// Randomly generate edges with weights between 1 and 10
// Set the diagonal to 0 and non-edges to a large value (infinity)
int **create_adjacency_matrix(int num_vertices) {
    srand(2468); // set seed for reproducibility
    int **matrix = (int **)malloc(num_vertices * sizeof(int *));
    for (int i = 0; i < num_vertices; i++) {
        matrix[i] = (int *)malloc(num_vertices * sizeof(int));
        for (int j = 0; j < num_vertices; j++) {
            if (i == j) {
                matrix[i][j] = 0;
            } else {
                // let there be a 70% chance of having an edge
                if ((rand() % 100) < 70) {
                    matrix[i][j] = rand() % 10 + 1; // random weight of 1-10
                } else {
                    matrix[i][j] = INF_EDGE; // no edge, set to "infinity" (10*num_vertices)
                }
            }
        }
    }
    return matrix;
}

// Free the adjacency matrix memory
void free_adjacency_matrix(int **matrix, int num_vertices) {
    for (int i = 0; i < num_vertices; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Print the adjacency matrix (for debugging purposes)
void print_graph(int **graph, int num_vertices) {
    for (int i = 0; i < num_vertices; i++) {
        for (int j = 0; j < num_vertices; j++) {
            if (graph[i][j] == INF_EDGE) {
                printf("INF ");
            } else {
                printf("%d ", graph[i][j]);
            }
        }
        printf("\n");
    }
}