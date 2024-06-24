import numpy as np
import spkmeans as sp
import sys
import pandas as pd

np.random.seed(0)

def init_first_centroid_randomly(data_points, k, indices_arr,data_points_indices):
    num_of_coordinates = data_points.shape[1]
    num_of_centroids = k
    centroids = np.zeros((num_of_centroids, num_of_coordinates))
    rnd_index = np.random.choice(data_points_indices)
    indices_arr.append(rnd_index)
    serial_index = data_points_indices.index(rnd_index)
    centroids[0] = data_points[serial_index]
    return centroids, indices_arr


def min_distance(data_points, k_centroids, num_of_centroids):
    num_of_vectors = data_points.shape[0]
    min_distances = []
    for i in range(num_of_vectors):
        min_dist = sys.float_info.max
        for j in range(num_of_centroids):
            dist = sum((data_points[i] - k_centroids[j]) ** 2)
            if dist < min_dist:
                min_dist = dist
        min_distances.append(min_dist)
    return min_distances


def probabilities(data_points, min_distances):
    num_of_vectors = data_points.shape[0]
    arr_probabilities = []
    sum_dist = sum(min_distances)
    for i in range(num_of_vectors):
        vector_min_dist = min_distances[i]
        if (sum_dist==0):
            print("An Error Has Occurred")
            sys.exit() 
        arr_probabilities.append(vector_min_dist / sum_dist)
    return arr_probabilities


def randomly_select_with_probability(data_points, arr_probabilities,indices_arr,data_points_indices):
    rnd_index = np.random.choice(data_points.shape[0], p=arr_probabilities)
    indices_arr.append(rnd_index)
    serial_index = data_points_indices.index(rnd_index)
    vector = data_points[serial_index]
    return vector,indices_arr


def init_k_centroids(k, data_points, data_points_indices):
    indices_arr = []
    centroids,indices_arr = init_first_centroid_randomly(data_points, k,indices_arr, data_points_indices)
    i = 1
    while i < k:
        distances = min_distance(data_points, centroids, i)
        arr_probabilities = probabilities(data_points, distances)
        centroids[i],indices_arr = randomly_select_with_probability(data_points, arr_probabilities,indices_arr,data_points_indices)
        i += 1
    return centroids, indices_arr

def print_matrix(matrix, N):
    for i in range(len(matrix)):
        matrix[i] = '%.4f' % matrix[i]
    for i in range(N):
        print(','.join(matrix[i*N:(i+1)*N])) 


def main(args):
    # initialize args from user
    # check num of args
    # Number of args == 3 (k, goal, file_name)
    if len(args) != 4:
        print("Invalid Input!")
        return

    # check k is a number
    k = args[1]
    try:
        k = int(args[1])
    except:
        print("Invalid Input!")
        return

    # check k >= 0
    if k < 0:
        print("Invalid Input!")
        return
    
    # file name (.txt or .csv): The path to the Input file, it will contain N data points
    try:
        input_file_name = str(args[3])
    except:
        print("Invalid Input!")
        return
    if (input_file_name[-4:] != ".txt") and (input_file_name[-4:] != ".csv"):
        print("Invalid Input!")
        return

    # reads input file into data points list
    data = pd.read_csv(input_file_name, header=None)
    data_points = data.to_numpy()

    N = data_points.shape[0]
    vector_len = data_points.shape[1]

    # data_points to array
    data_points_arr =[]
    N = data_points.shape[0]
    for i in range(N):
        for j in range(vector_len):
                data_points_arr.append(data_points[i][j])

    
    # check k<N
    if k >= N:
        print("Invalid Input!")
        return

    # goal (enum): Can get the following values
    goal = args[2]

    if goal == "wam":
        W_matrix = sp.capi_wam(vector_len, N, data_points_arr) 
        print_matrix(W_matrix,N) 

    elif goal == "ddg":
        D_matrix_arr = sp.capi_ddg(vector_len, N, data_points_arr) 
        print_matrix(D_matrix_arr,N) 

    elif goal == "lnorm":
        LNORM_matrix_arr = sp.capi_lnorm(vector_len, N, data_points_arr) 
        print_matrix(LNORM_matrix_arr,N)

    elif goal == "jacobi":
        eigen_matrix = sp.capi_jacobi(N, data_points_arr)
        
        # outputing eigenvalues & eigenvectors; the first line will be the eigenvalues, second line onward will be the corresponding eigenvectors
        for i in range(len(eigen_matrix)):
            eigen_matrix[i] = '%.4f' % eigen_matrix[i]
        for i in range(N+1):
            print(','.join(eigen_matrix[i*N:(i+1)*N]))

    elif goal == "spk":
        max_iter = 300  
        epsilon = 0 

        # gets T-matrix from C, as new data-points
        data_points = sp.capi_spk(k, N, vector_len, data_points_arr) # data_points_arr = T_matrix
        data_points = np.array(data_points)
        k = len(data_points[0]) # in case k == 0, heuristic updated k

        # py k-means (first centroids)
        centroids, indices_arr = init_k_centroids(k, data_points,[i for i in range(data_points.shape[0])])
        for i in range(centroids.shape[0]):
            for j in range(centroids.shape[1]):
                centroids[i][j] = '%.4f' % centroids[i][j]
        centroids = pd.DataFrame(centroids, columns=[i for i in range(centroids.shape[1])]).to_numpy()

        # outputing the initial centroids calculated in python
        for i in range(len(indices_arr)):
            indices_arr[i] = str(int(indices_arr[i]))
        print(','.join(indices_arr))

        # initial centroids to array
        centroids_arr =[]
        vector_len = centroids.shape[1]
        for i in range(k):
            for j in range(vector_len):
                    centroids_arr.append(centroids[i][j])

        # data_points to array
        data_points_arr =[]
        N = data_points.shape[0]
        for i in range(N):
            for j in range(k):
                data_points_arr.append(data_points[i][j])

        # interfacing with C extension - performing kmeanspp in C
        k_means_centroids = sp.capi_kmeans(k, max_iter, epsilon, data_points_arr, centroids_arr, k, N)
        
        # outputing the calculated final centroids from the K-means algorithm
        for i in range(len(k_means_centroids)):
            k_means_centroids[i] = '%.4f' % k_means_centroids[i]
        for i in range(k):
            print(','.join(k_means_centroids[i*vector_len:(i+1)*vector_len]))  


    elif ((goal != "jacobi") and  (goal != "wam") and (goal != "lnorm") and (goal != "ddg") and (goal != "spk")):
        print("Invalid Input!")
        return
    
# MAIN
main(sys.argv)
