#include "spkmeans.h"

/* MAIN */
int main(int argc, char *argv[])
{
    int vector_len, N, file_name_length;
    int *lengths_array;
    double *data_points, *W_matrix, *D_matrix, *LNORM_matrix, *eigen;
    char *goal, *filename;

    /* Reading user CMD arguments: (a)goal (b)file name */
    assertinput(argc == 3); /* Check number of arguments */
    goal = argv[1];         /* First argument from user - goal (enum) */
    assertinput((strcmp(goal, "wam") == 0) || strcmp(goal, "ddg") == 0 || strcmp(goal, "lnorm") == 0 || strcmp(goal, "jacobi") == 0);
    filename = argv[2]; /* Second argument from user - file name (.txt or .csv) */
    file_name_length = strlen(filename);
    assertinput(file_name_length >= 4);
    assertinput((strcmp(&filename[file_name_length - 4], ".txt") == 0) || (strcmp(&filename[file_name_length - 4], ".csv") == 0));

    /* Reading input file - for all goals except Jacobi: N data points, for Jacobi: a symmetric matrix */
    lengths_array = calc_lengths(filename);
    vector_len = lengths_array[0];
    N = lengths_array[1];
    free(lengths_array);
    data_points = calloc(N * vector_len, sizeof(double));
    assertruntime(data_points != NULL);
    read_input_file(filename, data_points);

    /* C Program (spkmeans.c) */
    if (strcmp(goal, "wam") == 0)
    {
        /* Calculate and output the Weighted Adjacency Matrix as described in 1.1.1 */
        W_matrix = calloc(N * N, sizeof(double));
        assertruntime(W_matrix);
        wam(data_points, W_matrix, N, vector_len);
        free(data_points);
        print_output(W_matrix, N, N);
        free(W_matrix);
    }

    else if (strcmp(goal, "ddg") == 0)
    {
        /* Calculate and output the Diagonal Degree Matrix as described in 1.1.2 */
        W_matrix = calloc(N * N, sizeof(double));
        assertruntime(W_matrix);
        wam(data_points, W_matrix, N, vector_len);
        free(data_points);

        D_matrix = calloc(N * N, sizeof(double));
        assertruntime(D_matrix);
        ddg(W_matrix, D_matrix, N);

        print_output(D_matrix, N, N);
        free(W_matrix);
        free(D_matrix);
    }

    else if (strcmp(goal, "lnorm") == 0)
    {
        /* Calculate and output the Normalized Graph Laplacian as described in 1.1.3 */
        W_matrix = calloc(N * N, sizeof(double));
        assertruntime(W_matrix);
        wam(data_points, W_matrix, N, vector_len);
        free(data_points);

        D_matrix = calloc(N * N, sizeof(double));
        assertruntime(D_matrix);
        ddg(W_matrix, D_matrix, N);

        LNORM_matrix = calloc(N * N, sizeof(double));
        assertruntime(LNORM_matrix);
        lnorm(W_matrix, D_matrix, LNORM_matrix, N);

        print_output(LNORM_matrix, N, N);
        free(W_matrix);
        free(D_matrix);
        free(LNORM_matrix);
    }

    else /*if (strcmp(goal, "jacobi") == 0) */
    {
        eigen = calloc((N + 1) * N, sizeof(double));
        Jacobi_algorithm(data_points, eigen, N);
        free(data_points);
        print_output(eigen, N + 1, N);
        free(eigen);
    }

    return 0;
}

/* Returns an array: [vector length, num of vectors] */
int *calc_lengths(char *input_filename)
{
    const char COMMA = ',';
    const char NEWLINE = '\n';
    int first_line = 0;
    int c = 0;
    FILE *f;

    /* Initialize an array of lengths: [vector_length, num of vectors] */
    int *lengths = calloc(2, sizeof(int));
    assertruntime(lengths != NULL);
    lengths[0] = 1;

    f = fopen(input_filename, "r");
    assertruntime(f != NULL);
    while ((c = fgetc(f)) != EOF)
    {
        if (first_line == 0 && c == COMMA)
        {
            lengths[0] += 1;
        }
        else if (c == NEWLINE)
        {
            lengths[1] += 1;
            if (first_line == 0)
            {
                first_line = 1;
            }
        }
    }
    fclose(f);
    return lengths;
}

/* Reading doubles coordinates seperated by commas and new lines into an array, and return it
 * N represents the amount of vectors */
void read_input_file(char *input_filename, double *data_points)
{
    /* allocate an array of size (num of vectors * len of vector) */
    int i = 0;
    double coordinate;
    int result;
    FILE *f;

    f = fopen(input_filename, "r");
    assertruntime(f != NULL);

    do
    {
        result = fscanf(f, "%lf", &coordinate);
        if (result == 0)
        {
            result = fscanf(f, "%*c");
        }
        else
        {
            data_points[i] = coordinate;
            i++;
        }

    } while (result != EOF);

    fclose(f);
}

/* WAM - 1.1.1 The Weighted Adjacency Matrix  */
void wam(double *coordinates, double *W_matrix, int N, int vector_len)
{
    double w_ij, vi_k_coordinate, vj_k_coordinate, norm;
    int i, j, k;

    /* We set wii = 0 for all i’s */
    for (i = 0; i < N; i++)
    {
        W_matrix[i * N + i] = 0;
    }

    /* The rest of the values are set to: wij = exp(-l2norm(xi − xj)/2) */
    for (i = 0; i < N; i++)
    {
        for (j = i + 1; j < N; j++) /* Symetric matrix */
        {
            /* Calculate W for vectors vi and vj */
            norm = 0;
            for (k = 0; k < vector_len; k++)
            {
                vi_k_coordinate = coordinates[i * vector_len + k];
                vj_k_coordinate = coordinates[j * vector_len + k];
                norm += pow(vi_k_coordinate - vj_k_coordinate, 2);
            }
            norm = sqrt(norm); /* The weights are non-negative (wij ≥ 0) */
            w_ij = exp(-(norm / 2));
            /* The weights are symmetric (wij = wji) */
            W_matrix[i * N + j] = w_ij;
            W_matrix[j * N + i] = w_ij;
        }
    }
}

/* DDG - 1.1.2 The Diagonal Degree Matrix */
void ddg(double *W_matrix, double *D_matrix, int N)
{
    double sum;
    int i, z;

    /* D matrix is zero, except for the diagonal */

    /* d_ij = if i = j: sum w_iz for 0 < z < N; otherwise: 0*/
    for (i = 0; i < N; i++)
    {
        sum = 0;
        for (z = 0; z < N; z++)
        {
            sum += W_matrix[i * N + z];
        }
        D_matrix[i * N + i] = sum;
    }
}

/* LNORM = 1.1.3 The Normalized Graph Laplacian */
void lnorm(double *W_matrix, double *D_matrix, double *LNORM_matrix, int N)
{
    int i = 0;
    double *temp, *I_matrix, *Mult_matrices_1, *Mult_matrices_2;

    /* Set I matrix */
    I_matrix = calloc(N * N, sizeof(double));
    assertruntime(I_matrix != NULL);
    for (i = 0; i < N; i++)
    {
        I_matrix[i * N + i] = 1;
    }

    /* Pow D by -1/2 */
    for (i = 0; i < N; i++)
    {
        D_matrix[i * N + i] = pow(D_matrix[i * N + i], -(0.5));
    }

    Mult_matrices_1 = mult_matrices(W_matrix, D_matrix, N);
    Mult_matrices_2 = mult_matrices(D_matrix, Mult_matrices_1, N);
    temp = sub_matrices(I_matrix, Mult_matrices_2, N);

    for (i = 0; i < N * N; i++)
    {
        LNORM_matrix[i] = temp[i];
    }

    free(temp);
    free(I_matrix);
    free(Mult_matrices_1);
    free(Mult_matrices_2);
}

/* Calculating matrix1 * matrix2, represented as arrays */
double *mult_matrices(double *matrix1, double *matrix2, int N)
{
    int i, j, k;
    double sum;
    double *mult = calloc(N * N, sizeof(double));
    assertruntime(mult != NULL);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            /* Calculating the i,j item */
            sum = 0;
            for (k = 0; k < N; k++)
            {
                sum += matrix1[i * N + k] * matrix2[k * N + j];
            }
            mult[i * N + j] = sum;
        }
    }
    return mult;
}

/* Calculating matrix1 - matrix2, represented as arrays */
double *sub_matrices(double *matrix1, double *matrix2, int N)
{
    int i, j;
    double *sub = calloc(N * N, sizeof(double));
    assertruntime(sub != NULL);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            sub[i * N + j] = matrix1[i * N + j] - matrix2[i * N + j];
        }
    }
    return sub;
}

/* Prints N*N matrix */
void print_output(double *output, int r, int c)
{
    int i, j;
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            printf("%.4f", output[i * c + j]);
            if (j < c - 1)
            {
                printf("%c", ',');
            }
        }
        printf("%c", '\n');
    }
}

/* Sign function - reurns 1 for >= 0, -1 for < 0 */
int sign(double theta)
{
    if (theta >= 0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

/* Jacobi algorithm -  an iterative method for the calculation of the eigenvalues and eigenvectors of a real symmetric matrixn*/
void Jacobi_algorithm(double *A_matrix, double *eigen, int N)
{
    int find_i = 0, find_j = 0, i = 0, j = 0, l = 0, count_iter = 0;
    double t, c, s, offA, offAtag, diff = 1, theta, sum, max = 0;
    double *V_matrix, *P_matrix, *P_trans, *temp_mult_metrix, *temp_mult_metrix1;

    /* V,P matrices init to I-matrix */
    V_matrix = calloc(N * N, sizeof(double));
    assertruntime(V_matrix != NULL);
    P_matrix = calloc(N * N, sizeof(double));
    assertruntime(P_matrix != NULL);
    temp_mult_metrix = calloc(N * N, sizeof(double));
    assertruntime(temp_mult_metrix != NULL);
    temp_mult_metrix1 = calloc(N * N, sizeof(double));
    assertruntime(temp_mult_metrix1 != NULL);

    for (i = 0; i < N; i++)
    {
        V_matrix[i * N + i] = 1;
        P_matrix[i * N + i] = 1;
    }

    while (count_iter < 100 && diff > 0.00001)
    {
        /* Calculate off(A)^2 */
        offA = 0;
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (j != i)
                {
                    offA += pow(A_matrix[i * N + j], 2);
                }
            }
        }
        /* If matrix is diag */
        if (offA == 0 && count_iter == 0)
        {
            break;
        }

        /* Find pivot i,j */
        max = 0;
        find_i = 0;
        find_j = 0;
        for (i = 0; i < N; i++)
        {
            for (j = i + 1; j < N; j++)
            {
                if (fabs(A_matrix[i * N + j]) > max)
                {
                    find_i = i;
                    find_j = j;
                    max = fabs(A_matrix[i * N + j]);
                }
            }
        }

        /* Obtain c,s */
        theta = (A_matrix[find_j * N + find_j] - A_matrix[find_i * N + find_i]) / (2 * A_matrix[find_i * N + find_j]);
        t = sign(theta) / (fabs(theta) + sqrt(pow(theta, 2) + 1));
        c = 1 / sqrt(pow(t, 2) + 1);
        s = t * c;

        /* Build a rotation matrix P Iinit to I-matrix and put c,s in the approp locs */
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (i == j)
                {
                    P_matrix[i * N + j] = 1;
                }
                else
                {
                    P_matrix[i * N + j] = 0;
                }
            }
        }
        P_matrix[find_i * N + find_i] = c;
        P_matrix[find_j * N + find_j] = c;
        P_matrix[find_i * N + find_j] = s;
        P_matrix[find_j * N + find_i] = -s;

        /* calculate V = P1P2P3... */
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                sum = 0;
                for (l = 0; l < N; l++)
                {
                    sum += V_matrix[i * N + l] * P_matrix[l * N + j];
                }
                temp_mult_metrix[i * N + j] = sum;
            }
        }
        for (i = 0; i < N * N; i++)
        {
            V_matrix[i] = temp_mult_metrix[i];
            temp_mult_metrix[i] = 0;
        }

        /* A`=Ptrans*A*P, A = A` */
        P_trans = transposed(P_matrix, N);

        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                sum = 0;
                for (l = 0; l < N; l++)
                {
                    sum += P_trans[i * N + l] * A_matrix[l * N + j];
                }
                temp_mult_metrix[i * N + j] = sum;
            }
        }
        free(P_trans);

        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                sum = 0;
                for (l = 0; l < N; l++)
                {
                    sum += temp_mult_metrix[i * N + l] * P_matrix[l * N + j];
                }
                temp_mult_metrix1[i * N + j] = sum;
            }
        }

        for (i = 0; i < N * N; i++)
        {
            A_matrix[i] = temp_mult_metrix1[i];
            temp_mult_metrix[i] = 0;
            temp_mult_metrix1[i] = 0;
        }

        /* Calculate off(A)^2 */
        offAtag = 0;
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (i != j)
                {
                    offAtag += pow(A_matrix[i * N + j], 2);
                }
            }
        }

        /* Calculate convergence and update iterations counter*/
        diff = offA - offAtag;
        count_iter++;
    }

    /* Output (eigen) is a (N+1)*N matrix: 1st row - eigenVals (diag of A-matrix), 2nd onwards - eigenVecs (V-matrix) */
    for (i = 0; i < (N + 1); i++)
    {
        for (j = 0; j < N; j++)
        {
            if (i == 0)
            {
                eigen[j] = A_matrix[j * N + j];
            }
            else
            {
                eigen[i * N + j] = V_matrix[(i - 1) * N + j];
            }
        }
    }

    free(P_matrix);
    free(V_matrix);
    free(temp_mult_metrix);
    free(temp_mult_metrix1);
}

/* Transpose */
double *transposed(double *matrix, int N)
{
    int i, j;
    double *transposed_matrix;
    transposed_matrix = calloc(N * N, sizeof(double));
    assertruntime(transposed_matrix != NULL);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            transposed_matrix[i * N + j] = matrix[j * N + i];
        }
    }
    return transposed_matrix;
}

/* Returns eigenvalues & eigenvectors in decrease order */
void decrease_order(double *eigen, int N, int vector_len)
{
    int i, j, k;
    double tmp_eigenvalue, *tmp_eigenvector, *eigenvalues, *eigenvectors;
    eigenvalues = calloc(N, sizeof(double));
    assertruntime(eigenvalues != NULL);
    eigenvectors = calloc(N * N, sizeof(double));
    assertruntime(eigenvectors != NULL);

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (i == 0)
            {
                eigenvalues[j] = eigen[j];
            }
            else
            {
                eigenvectors[(i - 1) * N + j] = eigen[i * N + j];
            }
        }
    }

    for (i = 0; i < N - 1; i++)
    {
        for (j = 0; j < N - i - 1; j++)
        {
            if (eigenvalues[j] < eigenvalues[j + 1])
            {
                tmp_eigenvalue = eigenvalues[j];
                eigenvalues[j] = eigenvalues[j + 1];
                eigenvalues[j + 1] = tmp_eigenvalue;

                tmp_eigenvector = calloc(vector_len, sizeof(double));
                assertruntime(tmp_eigenvector != NULL);
                for (k = 0; k < vector_len; k++)
                {
                    tmp_eigenvector[k] = eigenvectors[j * N + k];
                    eigenvectors[j * vector_len + k] = eigenvectors[(j + 1) * vector_len + k];
                    eigenvectors[(j + 1) * vector_len + k] = tmp_eigenvector[k];
                }
                free(tmp_eigenvector);
            }
        }
    }

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (i == 0)
            {
                eigen[j] = eigenvalues[j];
            }
            else
            {
                eigen[i * N + j] = eigenvalues[(i - 1) * N + j];
            }
        }
    }

    free(eigenvalues);
    free(eigenvectors);
}

/* The Eigengap Heuristic - In order to determine the number of clusters k, we will use eigengap heuristic: k = argmaxi(δi), i = 1,...,n/2, δi = |λi − λi+1| */
int determine_k_Eigengap_Heuristic(double *ordered_eigenvalues, int N)
{
    double max = 0;
    double *eigengaps;
    int i, argmaxi = 0;

    eigengaps = calloc(N, sizeof(double));
    ;
    assertruntime(eigengaps != NULL);

    for (i = 0; i < N - 1; i++)
    {
        eigengaps[i] = ordered_eigenvalues[i + 1] - ordered_eigenvalues[i];
    }

    for (i = 0; i < floor(N / 2); i++)
    {
        if (eigengaps[i] > max)
        {
            argmaxi = i;
            max = eigengaps[i];
        }
    }

    free(eigengaps);
    return argmaxi + 1;
}

/* The Normalized Spectral Clustering algorithm */
double *normalized_spectral_clustering_algorithm(double *coordinates, int N, int vector_len, int k)
{
    int i, j;
    double sum_row;
    double *temp, *sqrt_sum_rows_of_U, *W_matrix, *D_matrix, *LNORM_matrix, *eigen, *eigenvalues, *eigenvectors, *U_matrix, *T_matrix;

    /* Form the weighted adjacency matrix W from X */
    W_matrix = calloc(N * N, sizeof(double));
    assertruntime(W_matrix);
    wam(coordinates, W_matrix, N, vector_len);

    /* Compute the normalized graph Laplacian Lnorm */
    D_matrix = calloc(N * N, sizeof(double));
    assertruntime(D_matrix);
    ddg(W_matrix, D_matrix, N);

    LNORM_matrix = calloc(N * N, sizeof(double));
    assertruntime(LNORM_matrix);
    lnorm(W_matrix, D_matrix, LNORM_matrix, N);

    free(W_matrix);
    free(D_matrix);

    /* Determine k and obtain the largest k eigenvectors u1, . . . , uk of Lnorm */
    eigen = calloc((N + 1) * N, sizeof(double));
    assertruntime(eigen != NULL);

    Jacobi_algorithm(LNORM_matrix, eigen, N);
    free(LNORM_matrix);

    decrease_order(eigen, N, vector_len);

    eigenvalues = calloc(N, sizeof(double));
    assertruntime(eigenvalues != NULL);
    eigenvectors = calloc(N * N, sizeof(double));
    assertruntime(eigenvectors != NULL);

    for (i = 0; i < N; i++)
    {
        eigenvalues[i] = eigen[i];
    }

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            eigenvectors[i * N + j] = eigen[(i + 1) * N + j];
        }
    }

    free(eigen);

    if (k == 0)
    {
        k = determine_k_Eigengap_Heuristic(eigenvalues, N);
    }
    free(eigenvalues);

    /* Let U ∈ Rn×k be the matrix containing the vectors u1, . . . , uk as columns */
    temp = transposed(eigenvectors, N);
    free(eigenvectors);

    U_matrix = calloc(k * N, sizeof(double));
    assertruntime(U_matrix != NULL);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < k; j++)
        {
            U_matrix[i * k + j] = eigenvectors[i * N + j];
        }
    }
    free(temp);

    /* Form the matrix T ∈ Rn×k from U by renormalizing each of U’s rows */
    T_matrix = calloc(N * k, sizeof(double));
    assertruntime(T_matrix != NULL);

    sqrt_sum_rows_of_U = calloc(N, sizeof(double));
    assertruntime(sqrt_sum_rows_of_U != NULL);
    for (i = 0; i < N; i++)
    {
        sum_row = 0;
        for (j = 0; j < k; j++)
        {
            sum_row += pow(U_matrix[i * k + j], 2);
        }
        sqrt_sum_rows_of_U[i] = sqrt(sum_row);
    }

    /* Calc T from sqrt-row-sums of U & from U */
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < k; j++)
        {
            T_matrix[i * k + j] = U_matrix[i * k + j] / sqrt_sum_rows_of_U[i];
        }
    }

    free(U_matrix);
    free(sqrt_sum_rows_of_U);

    return T_matrix;
}

/* HW2 FROM HERE AND ONWARDS */
/* Kmeans alg */
void K_means(int K, int max_iter, double epsilon, double *coordinates, double *centroids, int N, int vector_len)
/*int main(int argc, char *argv[])*/
{
    int i;

    /* repeat until: convergence: ||∆µk|| < epsilon) -> "break" OR: iteration number = max iter -> for loop */
    for (i = 0; i < max_iter; i++)
    {
        double delta;
        int j;

        /* assign xi to the closest cluster Sj : argmin Sj (xi − µj)**2, ∀j 1 ≤ j ≤ K */
        int *closest_clusters_indeces = get_closest_clusters_indeces(centroids, coordinates, N, vector_len, K);

        /* for µk, 0 < k ≤ K: Update the centroids µk = (sum(x in Sk))/|Sk| */
        double *updated_centroids = update_centroids(K, N, vector_len, closest_clusters_indeces, coordinates);

        free(closest_clusters_indeces);

        /* if ||∆µk|| < epsilon then break */
        delta = 0;
        for (j = 0; j < K; j++)
        {
            double j_dist = distance(&centroids[j * vector_len], &updated_centroids[j * vector_len], vector_len);
            if (j_dist > delta)
            {
                delta = j_dist;
            }
        }

        /* update centroids */
        free(centroids);
        centroids = updated_centroids;

        if (delta < epsilon)
        {
            break;
        }
    }

    /* write k_means output to output file
    write_centroids(output_filename, vector_len, centroids, K);*/
    return;
}

/* Returns an array of size N (num of vectors), arr[i] = index of closest cluster for xi */
int *get_closest_clusters_indeces(double *centroids, double *coordinates, int N, int vector_len, int K)
{
    double *vector;
    double min_dist;
    int closest_index;
    double dist;
    int i;

    int *closest_clusters_index = calloc(N, sizeof(int));
    assertruntime(closest_clusters_index != NULL);

    for (i = 0; i < N; i++)
    {
        int j;
        vector = &coordinates[i * vector_len];
        min_dist = DBL_MAX;
        closest_index = 0;
        for (j = 0; j < K; j++)
        {
            dist = distance(vector, &centroids[j * vector_len], vector_len);
            if (dist < min_dist)
            {
                min_dist = dist;
                closest_index = j;
            }
        }
        closest_clusters_index[i] = closest_index;
    }
    return closest_clusters_index;
}

/* Returns the euclidean-dist between two vectors */
double distance(double *vec_1, double *vec_2, int vector_len)
{
    double curr_sum = 0;
    int i;
    for (i = 0; i < vector_len; i++) /* for each coordinate */
    {
        curr_sum = curr_sum + pow(*(vec_1 + i) - *(vec_2 + i), 2);
    }
    return sqrt(curr_sum);
}

/* Returns updated centroids */
double *update_centroids(int K, int N, int vector_len, int *closest_clusters_indeces, double *coordinates)
{
    double *new_centroids;
    int *cluster_sizes;
    int closest_cluster_index;
    int i;

    new_centroids = calloc(K * vector_len, sizeof(double));
    assertruntime(new_centroids != NULL);
    cluster_sizes = calloc(K, sizeof(int));
    assertruntime(cluster_sizes != NULL);

    /* sum vectors into clusters */
    for (i = 0; i < N; i++) /* for every vector */
    {
        int j;
        closest_cluster_index = closest_clusters_indeces[i];
        for (j = 0; j < vector_len; j++) /* for every coordinate in vector */
        {
            new_centroids[closest_cluster_index * vector_len + j] += coordinates[i * vector_len + j];
        }
        cluster_sizes[closest_cluster_index] += 1;
    }

    /* divide cluster by size */
    for (i = 0; i < K; i++)
    {
        int j;
        for (j = 0; j < vector_len; j++) /* for every coordinate-sum in cluster */
        {
            assertruntime(cluster_sizes[i] != 0);
            new_centroids[i * vector_len + j] /= cluster_sizes[i];
        }
    }

    free(cluster_sizes);

    return new_centroids;
}