/* 
EC527 Final Project
Scalar optimizations for Floyd-Warshall Algorithm
- Finds the All-Pairs-Shortest-Paths (APSP) of a randomly generated directed, weighted graph of multiple sizes (represented by adjacency matrices)
- Records the number of cycles taken to run the algorithm for each size

Time measurement code is borrowed from previous EC527 labs.

--------------------------------------------------------------------------------
gcc -O1 -mavx2 fw_scalar_optimizations.c -o fw_scalar_optimizations -lrt

*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#define A  64 /* coefficient of x^2 */
#define B  64  /* coefficient of x */
#define C  64  /* constant term */

#define NUM_TESTS 5// set to 8

#define CPNS 3.0

#define OPTIONS 10

#define IDENT 0

#define INF_EDGE 999999 // arbitrarily large value for infinity edges

#define BLOCK_SIZE 64

/* =================== Function Prototypes =================== */
int clock_gettime(clockid_t clk_id, struct timespec *tp);
int *create_adjacency_matrix(int num_vertices);
void free_adjacency_matrix(int *matrix, int num_vertices);
void print_graph(int *graph, int num_vertices);
void fw_serial(int *graph, int num_vertices);
void fw_local_variables(int *graph, int num_vertices);
void fw_loop_unroll2(int *graph, int num_vertices);
void fw_loop_unroll4(int *graph, int num_vertices);
void fw_loop_unroll8(int *graph, int num_vertices);
void fw_blocked(int *graph, int num_vertices);
void process_block(int *graph, int num_vertices, int i, int j, int k);
void process_block_lvars(int *graph, int num_vertices, int i, int j, int k);
void fw_blocked_unroll4(int *graph, int num_vertices);
void process_block_unroll4(int *graph, int num_vertices, int i, int j, int k);
void fw_simd_sse(int *graph, int num_vertices);
void fw_simd_avx(int *graph, int num_vertices);
void fw_blocked_unroll4_avx(int *graph, int num_vertices);
void process_block_unroll4_avx(int *graph, int num_vertices, int i, int j, int k);

/* =================== Time Measurement =================== */

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

/* =================== Wakeup Delay =================== */

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
int main() {
    int OPTION;
    int num_vertices, max_vertices;
    struct timespec time_start, time_stop;
    double time_stamp[OPTIONS][NUM_TESTS];
    double wd;
    int *graph;
    int *ref_graph[NUM_TESTS];
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
                // case 5: // unrolled implementation by factor of 4 with local variables
                //     // fw_loop_unroll4_lvars(graph, num_vertices); 
                //     break;
                case 5: // blocked implementation
                    fw_blocked(graph, num_vertices); 
                    break;
                case 6: // blocked implementation with unrolling by factor of 4 on inner loop
                    fw_blocked_unroll4(graph, num_vertices); 
                    break;
                case 7: // SIMD implementation w/ SSE intrinsics
                    fw_simd_sse(graph, num_vertices);
                    break;
                case 8: // SIMD implementation w/ AVX intrinsics
                    fw_simd_avx(graph, num_vertices);
                    break;
                case 9: // blocked implementation with unrolling by factor of 4 on inner loop using AVX intrinsics
                    fw_blocked_unroll4_avx(graph, num_vertices); 
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
                ref_graph[x] = (int *)malloc(num_vertices * num_vertices * sizeof(int *));
                for (int i = 0; i < num_vertices; i++) {
                    for (int j = 0; j < num_vertices; j++) {
                        ref_graph[x][i*num_vertices+j] = graph[i*num_vertices+j];
                    }
                }
            }

            // check if the results are correct (based on results from serial implementation)
            if (OPTION != 0) {
                for (int i = 0; i < num_vertices; i++) {
                    for (int j = 0; j < num_vertices; j++) {
                        if (graph[i*num_vertices+j] != ref_graph[x][i*num_vertices+j]) {
                            printf("Error: Results do not match for option %d at (%d, %d)\n", OPTION, i, j);
                            time_stamp[OPTION][x] = 0;
                            break;
                        }
                    }
                }
            }

            printf("  iter %d done\r", x); fflush(stdout);

            // free the adjacency matrix memory
            free_adjacency_matrix(graph, num_vertices);
        }
    }

    printf("\nnum_vertices, baseline, local variables, unroll 2x, unroll 4x, unroll 8x, blocked, blocked w/ unroll 4x, SIMD w/ SSE, SIMD w/ AVX2, blocked w/ unroll 4x & AVX2 \n");
    for (x = 0; x < NUM_TESTS && (num_vertices = A*x*x + B*x + C, num_vertices <= max_vertices); x++) {
        printf("%d", num_vertices);
        for (OPTION = 0; OPTION < OPTIONS; OPTION++) {
            printf(", %ld", (long int)((double)(CPNS) * 1.0e9 * time_stamp[OPTION][x]));
        }
        printf("\n");
    }

    for (x = 0; x < NUM_TESTS; x++) {
        if (ref_graph[x] != NULL) {
            free(ref_graph[x]);
        }
    }    

    printf("\n");
    printf("Initial delay was calculating: %g \n", wd); 

    return 0;
}

/* =================== Function Definitions =================== */
void fw_serial(int *graph, int num_vertices) {
    int i, j, k;
    for (k = 0; k < num_vertices; k++) {
        for (i = 0; i < num_vertices; i++) {
            for (j = 0; j < num_vertices; j++) {
                if (graph[i*num_vertices+k] + graph[k*num_vertices+j] < graph[i*num_vertices+j]) {
                    graph[i*num_vertices+j] = graph[i*num_vertices+k] + graph[k*num_vertices+j];
                }
            }
        }
    }
}

void fw_local_variables(int *graph, int num_vertices) {
    int i, j, k;
    for (k = 0; k < num_vertices; k++) {
        for (i = 0; i < num_vertices; i++) {
            int ik = graph[i*num_vertices+k];
            for (j = 0; j < num_vertices; j++) {
                int sum = ik + graph[k*num_vertices+j];
                if (sum < graph[i*num_vertices+j]) {
                    graph[i*num_vertices+j] = sum;
                }
            }
        }
    }
}

void fw_loop_unroll2(int *graph, int num_vertices) {
    int i, j, k;
    // unroll the innermost loop by a factor of 2
    for (k = 0; k < num_vertices; k++) {
        for (i = 0; i < num_vertices; i++) {
            int ik = graph[i*num_vertices+k];
            for (j = 0; j < num_vertices; j += 2) {
                int sum1 = ik + graph[k*num_vertices+j];
                int sum2 = ik + graph[k*num_vertices+j+1];
                if (sum1 < graph[i*num_vertices+j]) {
                    graph[i*num_vertices+j] = sum1;
                }
                if (j + 1 < num_vertices && sum2 < graph[i*num_vertices+j+1]) {
                    graph[i*num_vertices+j+1] = sum2;
                }
            }
        }
    }
}

void fw_loop_unroll4(int *graph, int num_vertices) {
    int i, j, k;
    // unroll the innermost loop by a factor of 4
    for (k = 0; k < num_vertices; k++) {
        for (i = 0; i < num_vertices; i++) {
            int ik = graph[i*num_vertices+k];
            for (j = 0; j < num_vertices; j += 4) {
                int sum1 = ik + graph[k*num_vertices+j];
                int sum2 = ik + graph[k*num_vertices+j+1];
                int sum3 = ik + graph[k*num_vertices+j+2];
                int sum4 = ik + graph[k*num_vertices+j+3];
                if (sum1 < graph[i*num_vertices+j]) {
                    graph[i*num_vertices+j] = sum1;
                }
                if (j + 1 < num_vertices && sum2 < graph[i*num_vertices+j+1]) {
                    graph[i*num_vertices+j+1] = sum2;
                }
                if (j + 2 < num_vertices && sum3 < graph[i*num_vertices+j+2]) {
                    graph[i*num_vertices+j+2] = sum3;
                }
                if (j + 3 < num_vertices && sum4 < graph[i*num_vertices+j+3]) {
                    graph[i*num_vertices+j+3] = sum4;
                }
            }
        }
    }
}

void fw_loop_unroll8(int *graph, int num_vertices) {
    int i, j, k;
    // unroll the innermost loop by a factor of 8
    for (k = 0; k < num_vertices; k++) {
        for (i = 0; i < num_vertices; i++) {
            int ik = graph[i*num_vertices+k];
            for (j = 0; j < num_vertices; j += 8) {
                int sum1 = ik + graph[k*num_vertices+j];
                int sum2 = ik + graph[k*num_vertices+j+1];
                int sum3 = ik + graph[k*num_vertices+j+2];
                int sum4 = ik + graph[k*num_vertices+j+3];
                int sum5 = ik + graph[k*num_vertices+j+4];
                int sum6 = ik + graph[k*num_vertices+j+5];
                int sum7 = ik + graph[k*num_vertices+j+6];
                int sum8 = ik + graph[k*num_vertices+j+7];
                if (sum1 < graph[i*num_vertices+j]) {
                    graph[i*num_vertices+j] = sum1;
                }
                if (j + 1 < num_vertices && sum2 < graph[i*num_vertices+j+1]) {
                    graph[i*num_vertices+j+1] = sum2;
                }
                if (j + 2 < num_vertices && sum3 < graph[i*num_vertices+j+2]) {
                    graph[i*num_vertices+j+2] = sum3;
                }
                if (j + 3 < num_vertices && sum4 < graph[i*num_vertices+j+3]) {
                    graph[i*num_vertices+j+3] = sum4;
                }
                if (j + 4 < num_vertices && sum5 < graph[i*num_vertices+j+4]) {
                    graph[i*num_vertices+j+4] = sum5;
                }
                if (j + 5 < num_vertices && sum6 < graph[i*num_vertices+j+5]) {
                    graph[i*num_vertices+j+5] = sum6;
                }
                if (j + 6 < num_vertices && sum7 < graph[i*num_vertices+j+6]) {
                    graph[i*num_vertices+j+6] = sum7;
                }
                if (j + 7 < num_vertices && sum8 < graph[i*num_vertices+j+7]) {
                    graph[i*num_vertices+j+7] = sum8;
                }
            }
        }
    }
}

// void fw_loop_unroll4_lvars(int *graph, int num_vertices) {
//     int i, j, k;
//     // unroll the innermost loop by a factor of 4 with local variables
//     for (k = 0; k < num_vertices; k++) {
//         for (i = 0; i < num_vertices; i++) {
//             int ik = graph[i*num_vertices+k];
//             for (j = 0; j < num_vertices; j += 4) {
//                 int sum1 = ik + graph[k*num_vertices+j];
//                 int sum2 = ik + graph[k*num_vertices+j+1];
//                 int sum3 = ik + graph[k*num_vertices+j+2];
//                 int sum4 = ik + graph[k*num_vertices+j+3];
//                 if (sum1 < graph[i*num_vertices+j]) {
//                     graph[i*num_vertices+j] = sum1;
//                 }
//                 if (j + 1 < num_vertices && sum2 < graph[i*num_vertices+j+1]) {
//                     graph[i*num_vertices+j+1] = sum2;
//                 }
//                 if (j + 2 < num_vertices && sum3 < graph[i*num_vertices+j+2]) {
//                     graph[i*num_vertices+j+2] = sum3;
//                 }
//                 if (j + 3 < num_vertices && sum4 < graph[i*num_vertices+j+3]) {
//                     graph[i*num_vertices+j+3] = sum4;
//                 }
//             }
//         }
//     }
// }

void fw_blocked(int *graph, int num_vertices) {
    int i, j, k;
    for (k = 0; k < num_vertices; k += BLOCK_SIZE) {
        // process the diagonal block
        process_block_lvars(graph, num_vertices, k, k, k);

        // process row and column blocks
        for (i = 0; i < num_vertices; i += BLOCK_SIZE) {
            if (i != k) {
                process_block_lvars(graph, num_vertices, i, k, k);
            }
        }

        for (j = 0; j < num_vertices; j += BLOCK_SIZE) {
            if (j != k) {
                process_block_lvars(graph, num_vertices, k, j, k);
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

// void process_block(int *graph, int num_vertices, int i, int j, int k) {
//     for (int kk = k; kk < k + BLOCK_SIZE && kk < num_vertices; kk++) {
//         for (int ii = i; ii < i + BLOCK_SIZE && ii < num_vertices; ii++) {
//             for (int jj = j; jj < j + BLOCK_SIZE && jj < num_vertices; jj++) {
//                 if (graph[ii*num_vertices+kk] + graph[kk*num_vertices+jj] < graph[ii*num_vertices+jj]) {
//                     graph[ii*num_vertices+jj] = graph[ii*num_vertices+kk] + graph[kk*num_vertices+jj];
//                 }
//             }
//         }
//     }
// }

void process_block_lvars(int *graph, int num_vertices, int i, int j, int k) {
    int ii, jj, kk;
    for (kk = k; kk < k + BLOCK_SIZE && kk < num_vertices; kk++) {
        for (ii = i; ii < i + BLOCK_SIZE && ii < num_vertices; ii++) {
            int ik = graph[ii*num_vertices+kk];
            for (jj = j; jj < j + BLOCK_SIZE && jj < num_vertices; jj++) {
                int sum = ik + graph[kk*num_vertices+jj];
                if (sum < graph[ii*num_vertices+jj]) {
                    graph[ii*num_vertices+jj] = sum;
                }
            }
        }
    }
}

void fw_blocked_unroll4(int *graph, int num_vertices) {
    int i, j, k;
    for (k = 0; k < num_vertices; k += BLOCK_SIZE) {
        // process the diagonal block
        process_block_unroll4(graph, num_vertices, k, k, k);

        // process row and column blocks
        for (i = 0; i < num_vertices; i += BLOCK_SIZE) {
            if (i != k) {
                process_block_unroll4(graph, num_vertices, i, k, k);
                process_block_unroll4(graph, num_vertices, k, i, k);
            }
        }

        // process the remaining blocks
        for (i = 0; i < num_vertices; i += BLOCK_SIZE) {
            if (i == k) continue;
            for (j = 0; j < num_vertices; j += BLOCK_SIZE) {
                if (j == k) continue;
                process_block_unroll4(graph, num_vertices, i, j, k);
            }
        }
    }
}

void process_block_unroll4(int *graph, int num_vertices, int i, int j, int k) {
    int ii, jj, kk;
    for (kk = k; kk < k + BLOCK_SIZE && kk < num_vertices; kk++) {
        for (ii = i; ii < i + BLOCK_SIZE && ii < num_vertices; ii++) {
            int ik = graph[ii*num_vertices+kk];
            for (jj = j; jj < j + BLOCK_SIZE && jj < num_vertices; jj += 4) {
                int sum1 = ik + graph[kk*num_vertices+jj];
                int sum2 = ik + graph[kk*num_vertices+jj+1];
                int sum3 = ik + graph[kk*num_vertices+jj+2];
                int sum4 = ik + graph[kk*num_vertices+jj+3];
                if (sum1 < graph[ii*num_vertices+jj]) {
                    graph[ii*num_vertices+jj] = sum1;
                }
                if (jj + 1 < num_vertices && sum2 < graph[ii*num_vertices+jj+1]) {
                    graph[ii*num_vertices+jj+1] = sum2;
                }
                if (jj + 2 < num_vertices && sum3 < graph[ii*num_vertices+jj+2]) {
                    graph[ii*num_vertices+jj+2] = sum3;
                }
                if (jj + 3 < num_vertices && sum4 < graph[ii*num_vertices+jj+3]) {
                    graph[ii*num_vertices+jj+3] = sum4;
                }
            }
        }
    }
}

void fw_simd_sse(int *graph, int num_vertices) {
    // using SSE intrinsics for SIMD operations
    int i, j, k;
    for (k = 0; k < num_vertices; k++) {
        for (i = 0; i < num_vertices; i++) {
            __m128i ik = _mm_set1_epi32(graph[i*num_vertices+k]);
            for (j = 0; j < num_vertices; j += 4) {
                __m128i ij = _mm_loadu_si128((__m128i *)&graph[i*num_vertices+j]);
                __m128i kj = _mm_loadu_si128((__m128i *)&graph[k*num_vertices+j]);
                __m128i sum = _mm_add_epi32(ik, kj);
                __m128i mask = _mm_cmpgt_epi32(ij, sum); // see is ij > sum, set mask = 1 if true
                __m128i result = _mm_or_si128(_mm_and_si128(mask, sum), _mm_andnot_si128(mask, ij)); // select between sum and ij based on mask
                _mm_storeu_si128((__m128i *)&graph[i*num_vertices+j], result);
            }
        }
    }
}

void fw_simd_avx(int *graph, int num_vertices) {
    // using AVX2 intrinsics for SIMD operations
    int i, j, k;
    for (k = 0; k < num_vertices; k++) {
        for (i = 0; i < num_vertices; i++) {
            __m256i ik = _mm256_set1_epi32(graph[i*num_vertices+k]);
            for (j = 0; j < num_vertices; j += 8) {
                __m256i ij = _mm256_loadu_si256((__m256i *)&graph[i*num_vertices+j]);
                __m256i kj = _mm256_loadu_si256((__m256i *)&graph[k*num_vertices+j]);
                __m256i sum = _mm256_add_epi32(ik, kj);
                __m256i mask = _mm256_cmpgt_epi32(ij, sum);
                __m256i result = _mm256_or_si256(_mm256_and_si256(mask, sum), _mm256_andnot_si256(mask, ij)); // select between sum and ij based on mask
                _mm256_storeu_si256((__m256i *)&graph[i*num_vertices+j], result);}
        }
    }
}

void fw_blocked_unroll4_avx(int *graph, int num_vertices) {
    int i, j, k;
    for (k = 0; k < num_vertices; k += BLOCK_SIZE) {
        // process the diagonal block
        process_block_unroll4_avx(graph, num_vertices, k, k, k);

        // process row and column blocks
        for (i = 0; i < num_vertices; i += BLOCK_SIZE) {
            if (i != k) {
                process_block_unroll4_avx(graph, num_vertices, i, k, k);
                process_block_unroll4_avx(graph, num_vertices, k, i, k);
            }
        }

        // process the remaining blocks
        for (i = 0; i < num_vertices; i += BLOCK_SIZE) {
            if (i == k) continue;
            for (j = 0; j < num_vertices; j += BLOCK_SIZE) {
                if (j == k) continue;
                process_block_unroll4_avx(graph, num_vertices, i, j, k);
            }
        }
    }
}

void process_block_unroll4_avx(int *graph, int num_vertices, int i, int j, int k) {
    int ii, jj, kk;
    for (kk = k; kk < k + BLOCK_SIZE && kk < num_vertices; kk++) {
        for (ii = i; ii < i + BLOCK_SIZE && ii < num_vertices; ii++) {
            __m256i ik = _mm256_set1_epi32(graph[ii*num_vertices+kk]);
            for (jj = j; jj < j + BLOCK_SIZE && jj < num_vertices; jj+= 16) {
                __m256i ij1 = _mm256_loadu_si256((__m256i *)&graph[ii*num_vertices+jj]);
                __m256i kj1 = _mm256_loadu_si256((__m256i *)&graph[kk*num_vertices+jj]);
                __m256i ij2 = _mm256_loadu_si256((__m256i *)&graph[ii*num_vertices+jj+8]);
                __m256i kj2 = _mm256_loadu_si256((__m256i *)&graph[kk*num_vertices+jj+8]);
                // __m256i ij3 = _mm256_loadu_si256((__m256i *)&graph[ii][jj + 16]);
                // __m256i kj3 = _mm256_loadu_si256((__m256i *)&graph[kk][jj + 16]);
                // __m256i ij4 = _mm256_loadu_si256((__m256i *)&graph[ii][jj + 24]);    
                // __m256i kj4 = _mm256_loadu_si256((__m256i *)&graph[kk][jj + 24]);


                __m256i sum1 = _mm256_add_epi32(ik, kj1);
                __m256i sum2 = _mm256_add_epi32(ik, kj2);
                // __m256i sum3 = _mm256_add_epi32(ik, kj3);
                // __m256i sum4 = _mm256_add_epi32(ik, kj4);

                __m256i mask1 = _mm256_cmpgt_epi32(ij1, sum1);
                __m256i mask2 = _mm256_cmpgt_epi32(ij2, sum2);
                // __m256i mask3 = _mm256_cmpgt_epi32(ij3, sum3);
                // __m256i mask4 = _mm256_cmpgt_epi32(ij4, sum4);

                __m256i result1 = _mm256_or_si256(_mm256_and_si256(mask1, sum1),_mm256_andnot_si256(mask1, ij1)); // select between sum and ij based on mask
                __m256i result2 = _mm256_or_si256(_mm256_and_si256(mask2, sum2), _mm256_andnot_si256(mask2, ij2)); // select between sum and ij based on mask
                // __m256i result3 = _mm256_or_si256(_mm256_and_si256(mask3, sum3), _mm256_andnot_si256(mask3, ij3)); // select between sum and ij based on mask 
                // __m256i result4 = _mm256_or_si256(_mm256_and_si256(mask4, sum4), _mm256_andnot_si256(mask4, ij4)); // select between sum and ij based on mask
                
                _mm256_storeu_si256((__m256i *)&graph[ii*num_vertices+jj], result1);
                _mm256_storeu_si256((__m256i *)&graph[ii*num_vertices+jj+8], result2);
                // _mm256_storeu_si256((__m256i *)&graph[ii][jj + 16], result3);
                // _mm256_storeu_si256((__m256i *)&graph[ii][jj + 24], result4);
            }
        }
    }
}

// Create an adjacency matrix for the graph
// Randomly generate edges with weights between 1 and 10
// Set the diagonal to 0 and non-edges to a large value (infinity)
int *create_adjacency_matrix(int num_vertices) {
    srand(2468); // set seed for reproducibility
    int *matrix = (int *)malloc(num_vertices * num_vertices * sizeof(int));
    for (int i = 0; i < num_vertices; i++) {
        for (int j = 0; j < num_vertices; j++) {
            if (i == j) {
                matrix[i*num_vertices+j] = 0;
            } else {
                // let there be a 70% chance of having an edge
                if ((rand() % 100) < 70) {
                    matrix[i*num_vertices+j] = rand() % 10 + 1; // random weight of 1-10
                } else {
                    matrix[i*num_vertices+j] = INF_EDGE; // no edge, set to "infinity" (10*num_vertices)
                }
            }
        }
    }
    return matrix;
}

// Free the adjacency matrix memory
void free_adjacency_matrix(int *matrix, int num_vertices) {
    free(matrix);
}

// Print the adjacency matrix (for debugging purposes)
void print_graph(int *graph, int num_vertices) {
    for (int i = 0; i < num_vertices; i++) {
        for (int j = 0; j < num_vertices; j++) {
            if (graph[i*num_vertices+j] == INF_EDGE) {
                printf("INF ");
            } else {
                printf("%d ", graph[i*num_vertices+j]);
            }
        }
        printf("\n");
    }
}
