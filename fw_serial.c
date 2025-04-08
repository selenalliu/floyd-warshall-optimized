/* 
EC527 Final Project
Serial reference code of the Floyd-Warshall Algorithm
- Finds the All-Pairs-Shortest-Paths (APSP) of a randomly generated directed, weighted graph of multiple sizes (represented by adjacency matrices)
- Records the time (ms) taken to run the algorithm for each size

Time measurement code is borrowed from previous EC527 labs.

--------------------------------------------------------------------------------
gcc -O0 -o fw_serial fw_serial.c -lrt

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h> // For INT_MAX

#define A  8  /* coefficient of x^2 */
#define B  16  /* coefficient of x */
#define C  32  /* constant term */

#define NUM_TESTS 15

#define OPTIONS 1

#define IDENT 0

typedef int data_t;

/* =================== Function Prototypes =================== */
int clock_gettime(clockid_t clk_id, struct timespec *tp);
void floyd_warshall_serial(int **graph, int num_vertices);
int **create_adjacency_matrix(int num_vertices);
void free_adjacency_matrix(int **matrix, int num_vertices);
// void print_graph(int **graph, int num_vertices);

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
    int num_vertices, max_vertices;
    struct timespec time_start, time_stop;
    double time_stamp[NUM_TESTS];
    double wd;
    int **graph;
    int x;

    printf("Floyd-Warshall Algorithm - Serial Implementation\n");

    wd = wakeup_delay();

    x = NUM_TESTS - 1;
    max_vertices = A*x*x + B*x + C;

    printf("\nAll times are in milliseconds\n");
    printf("num_vertices, time_taken\n");

    for (x = 0; x < NUM_TESTS && (num_vertices = A*x*x + B*x + C, num_vertices <= max_vertices); x++) {

        // create the adjacency matrix
        graph = create_adjacency_matrix(num_vertices);

        // start timing the algorithm
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
        
        // run the Floyd-Warshall algorithm
        floyd_warshall_serial(graph, num_vertices);
        
        // stop timing the algorithm
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
        
        // calculate and store the time taken
        time_stamp[x] = interval(time_start, time_stop) * 1000; // convert to milliseconds

        printf("%d, %f\n", num_vertices, time_stamp[x]);

        // Free the adjacency matrix memory
        free_adjacency_matrix(graph, num_vertices);
    }

    printf("\n");
    printf("Initial delay was calculating: %g \n", wd);

    return 0;

}

/* =================== Function Definitions =================== */
void floyd_warshall_serial(int **graph, int num_vertices) {
    int i, j, k;
    // Update the adjacency matrix with the shortest paths
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
                    matrix[i][j] = INT_MAX; // no edge, set to "infinity"
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
