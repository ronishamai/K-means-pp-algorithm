#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <ctype.h>

#define EpsilonJacobi 0.00001
#define Epsilon 0
#define MaxIter 300
#define MaxRotations 100

#define assertinput(expr)         \
    if (!(expr))                  \
    {                             \
        printf("Invalid Input!"); \
        exit(1);                  \
    }
#define assertruntime(expr)              \
    if (!(expr))                         \
    {                                    \
        printf("An Error Has Occurred"); \
        exit(1);                         \
    }

/* HEADERS */
/* Main functions */
int *calc_lengths(char *input_filename);
void read_input_file(char *filename, double *data_points);
void wam(double *coordinates, double *W_matrix, int N, int vector_len);
void ddg(double *W_matrix, double *D_matrix, int N);
void lnorm(double *W_matrix, double *D_matrix, double *LNORM_matrix, int N);
void Jacobi_algorithm(double *A_matrix, double *eigen, int N);
int determine_k_Eigengap_Heuristic(double *ordered_eigenvalues, int N);
double *normalized_spectral_clustering_algorithm(double *coordinates, int N, int vector_len, int k);

/* Helpers */
double *mult_matrices(double *matrix1, double *matrix2, int N);
double *sub_matrices(double *matrix1, double *matrix2, int N);
void print_output(double *output, int r, int c);
int sign(double theta);
void decrease_order(double *eigen, int N, int vector_len);
double *transposed(double *matrix, int N);

/* HW2 */
int *get_closest_clusters_indeces(double *centroids, double *coordinates, int N, int vector_len, int K);
double distance(double *vec_1, double *vec_2, int vector_len);
double *update_centroids(int K, int N, int vector_len, int *closest_clusters_indeces, double *coordinates);
void K_means(int K, int max_iter, double epsilon, double *coordinates, double *centroids, int N, int vector_len);