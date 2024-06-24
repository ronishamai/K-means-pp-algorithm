#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spkmeans.h"

/* Kmeans PY-C API as HW2 */
static PyObject *capi_kmeans(PyObject *self, PyObject *args)
{
    int i, j, k, N, vector_len, max_iter;
    double eps;
    double *data_points, *centroids;
    PyObject *data_points_py, *centroids_py, *updated_centroids_py;

    /* Check arguments */
    if (!PyArg_ParseTuple(args, "iidOOii", &k, &max_iter, &eps, &data_points_py, &centroids_py, &vector_len, &N))
    {
        return NULL;
    }

    /* C-Python conversions and K-MEANS on data_points & initial centroids (from python k-meanspp) */

    data_points = calloc(N * vector_len, sizeof(double));
    assertruntime(data_points != NULL);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < vector_len; j++)
        {
            data_points[i * vector_len + j] = PyFloat_AsDouble(PyList_GetItem(data_points_py, i * vector_len + j));
        }
    }

    centroids = calloc(k * vector_len, sizeof(double));
    assertruntime(centroids != NULL);
    for (i = 0; i < k; i++)
    {
        for (j = 0; j < vector_len; j++)
        {
            centroids[i * vector_len + j] = PyFloat_AsDouble(PyList_GetItem(centroids_py, i * vector_len + j));
        }
    }

    K_means(k, max_iter, eps, data_points, centroids, N, vector_len);

    updated_centroids_py = PyList_New(k * vector_len);
    for (i = 0; i < k; i++)
    {
        for (j = 0; j < vector_len; j++)
        {
            PyList_SetItem(updated_centroids_py, i * vector_len + j, PyFloat_FromDouble(centroids[i * vector_len + j]));
        }
    }

    free(data_points);
    /* free(centroids); */
    return updated_centroids_py;
}

/* C-PY API: Perform full spectral kmeans as described in 1 */
static PyObject *capi_spk(PyObject *self, PyObject *args)
{
    PyObject *data_points_py, *T_matrix_py;
    double *data_points, *T_matrix;
    int i, j, N, k, vector_len;

    if (!PyArg_ParseTuple(args, "iO", &k, &N, &vector_len, &data_points_py))
    {
        return NULL;
    }

    data_points = calloc(N * N, sizeof(double));
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            data_points[i * N + j] = PyFloat_AsDouble(PyList_GetItem(data_points_py, i * N + j));
        }
    }

    T_matrix = calloc((N + 1) * N, sizeof(double));
    normalized_spectral_clustering_algorithm(data_points, N, vector_len, k);
    k = (int)((int)(sizeof(T_matrix) / sizeof(double))) / N;

    T_matrix_py = PyList_New((N + 1) * N);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < k; j++)
        {
            PyList_SetItem(T_matrix_py, i * k + j, PyFloat_FromDouble(T_matrix[i * k + j]));
        }
    }

    free(T_matrix);
    free(data_points);
    return T_matrix_py;
    /* Perform K-MEANSPP on python */
}

/* C-PY API: Calculate and output the Weighted Adjacency Matrix as described in 1.1.1 */
static PyObject *capi_wam(PyObject *self, PyObject *args)
{
    PyObject *data_points_py, *W_matrix_py;
    double *data_points, *W_matrix;
    int i, j, N, vector_len;

    if (!PyArg_ParseTuple(args, "iiO", &vector_len, &N, &data_points_py))
    {
        return NULL;
    }

    data_points = calloc(N * vector_len, sizeof(double));
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < vector_len; j++)
        {
            data_points[i * vector_len + j] = PyFloat_AsDouble(PyList_GetItem(data_points_py, i * vector_len + j));
        }
    }

    W_matrix = calloc(N * N, sizeof(double));
    wam(data_points, W_matrix, N, vector_len);
    W_matrix_py = PyList_New(N * N);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            PyList_SetItem(W_matrix_py, i * N + j, PyFloat_FromDouble(W_matrix[i * N + j]));
        }
    }

    free(W_matrix);
    free(data_points);
    return W_matrix_py;
}

/* C-PY API: Calculate and output the Diagonal Degree Matrix as described in 1.1.2 */
static PyObject *capi_ddg(PyObject *self, PyObject *args)
{
    PyObject *data_points_py, *D_matrix_py;
    double *data_points, *W_matrix, *D_matrix;
    int i, j, N, vector_len;

    if (!PyArg_ParseTuple(args, "iiO", &vector_len, &N, &data_points_py))
    {
        return NULL;
    }

    data_points = calloc(N * vector_len, sizeof(double));
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < vector_len; j++)
        {
            data_points[i * vector_len + j] = PyFloat_AsDouble(PyList_GetItem(data_points_py, i * vector_len + j));
        }
    }

    W_matrix = calloc(N * N, sizeof(double));
    wam(data_points, W_matrix, N, vector_len);

    D_matrix = calloc(N * N, sizeof(double));
    ddg(W_matrix, D_matrix, N);

    D_matrix_py = PyList_New(N * N);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            PyList_SetItem(D_matrix_py, i * N + j, PyFloat_FromDouble(D_matrix[i * N + j]));
        }
    }

    free(W_matrix);
    free(D_matrix);
    free(data_points);
    return D_matrix_py;
}

/* C-PY API:  Calculate and output the Normalized Graph Laplacian as described in 1.1.3 */
static PyObject *capi_lnorm(PyObject *self, PyObject *args)
{
    PyObject *data_points_py, *LNORM_matrix_py;
    double *data_points, *W_matrix, *D_matrix, *LNORM_matrix;
    int i, j, N, vector_len;

    if (!PyArg_ParseTuple(args, "iiO", &vector_len, &N, &data_points_py))
    {
        return NULL;
    }

    data_points = calloc(N * vector_len, sizeof(double));
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < vector_len; j++)
        {
            data_points[i * vector_len + j] = PyFloat_AsDouble(PyList_GetItem(data_points_py, i * vector_len + j));
        }
    }

    W_matrix = calloc(N * N, sizeof(double));
    wam(data_points, W_matrix, N, vector_len);

    D_matrix = calloc(N * N, sizeof(double));
    ddg(W_matrix, D_matrix, N);

    LNORM_matrix = calloc(N * N, sizeof(double));
    lnorm(W_matrix, D_matrix, LNORM_matrix, N);

    LNORM_matrix_py = PyList_New(N * N);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            PyList_SetItem(LNORM_matrix_py, i * N + j, PyFloat_FromDouble(LNORM_matrix[i * N + j]));
        }
    }

    free(W_matrix);
    free(D_matrix);
    free(LNORM_matrix);
    free(data_points);
    return LNORM_matrix_py;
}

/* C-PY API: Calculate and output the eigenvalues and eigenvectors as described in 1.2.1 */
static PyObject *capi_jacobi(PyObject *self, PyObject *args)
{
    PyObject *sym_matrix_py, *EIGEN_matrix_py;
    double *sym_matrix, *EIGEN_matrix;
    int i, j, N;

    if (!PyArg_ParseTuple(args, "iO", &N, &sym_matrix_py))
    {
        return NULL;
    }

    sym_matrix = calloc(N * N, sizeof(double));
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            sym_matrix[i * N + j] = PyFloat_AsDouble(PyList_GetItem(sym_matrix_py, i * N + j));
        }
    }

    EIGEN_matrix = calloc((N + 1) * N, sizeof(double));
    Jacobi_algorithm(sym_matrix, EIGEN_matrix, N);

    EIGEN_matrix_py = PyList_New((N + 1) * N);
    for (i = 0; i < N + 1; i++)
    {
        for (j = 0; j < N; j++)
        {
            PyList_SetItem(EIGEN_matrix_py, i * N + j, PyFloat_FromDouble(EIGEN_matrix[i * N + j]));
        }
    }

    free(EIGEN_matrix);
    free(sym_matrix);
    return EIGEN_matrix_py;
}

/* C-PY API */
static PyMethodDef capiMethods[] = {
    {"capi_kmeans", (PyCFunction)capi_kmeans, METH_VARARGS, PyDoc_STR("returning k clusters")},
    {"capi_spk", (PyCFunction)capi_spk, METH_VARARGS, PyDoc_STR("preforming normalized spectral clustering algorithm")},
    {"capi_wam", (PyCFunction)capi_wam, METH_VARARGS, PyDoc_STR("returning W matrix")},
    {"capi_ddg", (PyCFunction)capi_ddg, METH_VARARGS, PyDoc_STR("returning D matrix")},
    {"capi_lnorm", (PyCFunction)capi_lnorm, METH_VARARGS, PyDoc_STR("returning LNORM matrix")},
    {"capi_jacobi", (PyCFunction)capi_jacobi, METH_VARARGS, PyDoc_STR("returning eigenvalues and eigenvectors")},
    {NULL, NULL, 0, NULL}};

/* C-PY API */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spkmeans",
    NULL,
    -1,
    capiMethods};

/* C-PY API */
PyMODINIT_FUNC
PyInit_spkmeans(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m)
    {
        return NULL;
    }
    return m;
}