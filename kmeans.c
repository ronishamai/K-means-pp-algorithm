#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <ctype.h>

 #define assertruntime(expr) \
 if (!(expr)) \
 { \
     printf("An Error Has Occurred"); \
     exit(1); \
 }

int *get_closest_clusters_indeces(double *centroids, double *coordinates, int N, int vector_len, int K);
double distance(double *vec_1, double *vec_2, int vector_len);
double *update_centroids(int K, int N, int vector_len, int *closest_clusters_indeces, double *coordinates);
void K_means(int K ,int max_iter, double epsilon, double *coordinates, double *centroids, int N, int vector_len);


static PyObject* fit(PyObject *self, PyObject *args)
{
    int k;
    int max_iter;
    double eps;
    PyObject *data_points_py;
    PyObject *centroids_py;
    int vector_len;
    int N;
    double *data_points;
    double *centroids;
    PyObject *updated_centroids_py;
    int i;
    int j;

    if(!PyArg_ParseTuple(args, "iidOOii", &k, &max_iter, &eps, &data_points_py, &centroids_py, &vector_len, &N)) {
        return NULL;
    }

    data_points = calloc(N*vector_len,sizeof(double));
    for (i = 0; i < N; i++){
        for (j = 0; j < vector_len; j++)
        {
            data_points[i*vector_len+j] = PyFloat_AsDouble(PyList_GetItem(data_points_py, i*vector_len+j));
        }
    }

    centroids = calloc(k*vector_len,sizeof(double));
    for (i = 0; i < k; i++){
        for (j = 0; j < vector_len; j++)
        {
            centroids[i*vector_len+j] = PyFloat_AsDouble(PyList_GetItem(centroids_py, i*vector_len+j));
        }
    }
    
    K_means(k, max_iter, eps, data_points, centroids, N, vector_len);

    updated_centroids_py = PyList_New(k*vector_len);
    for (i = 0; i < k; i++){
        for (j = 0; j < vector_len; j++)
        {
            PyList_SetItem(updated_centroids_py, i*vector_len+j,PyFloat_FromDouble(centroids[i*vector_len+j]));
        }
    }

    free(data_points);
    /* free(centroids); */

    return updated_centroids_py;
}

static PyMethodDef kmeansMethods[] = {
    {"fit",
        (PyCFunction) fit,
        METH_VARARGS,
        PyDoc_STR("returning k clusters")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    kmeansMethods
};

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}


void K_means(int K ,int max_iter, double epsilon, double *coordinates, double *centroids, int N, int vector_len)
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

/* returns an array of size N (num of vectors), arr[i] = index of closest cluster for xi */
int *get_closest_clusters_indeces(double *centroids, double *coordinates, int N, int vector_len, int K)
{
    double *vector;
    double min_dist;
    int closest_index;
    double dist;
    int i;

    int *closest_clusters_index = calloc(N, sizeof(int));
    assertruntime(closest_clusters_index!=NULL);

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

/* returns the euclidean-dist between two vectors */
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

double *update_centroids(int K, int N, int vector_len, int *closest_clusters_indeces, double *coordinates)
{
    double *new_centroids;
    int *cluster_sizes;
    int closest_cluster_index;
    int i;

    new_centroids = calloc(K * vector_len, sizeof(double));
    assertruntime(new_centroids!=NULL);
    cluster_sizes = calloc(K, sizeof(int));
    assertruntime(cluster_sizes!=NULL);    

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