import sys
import numpy as np
import pandas as pd
import mykmeanssp
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
    rnd_index = np.random.choice(data_points_indices, p=arr_probabilities)
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


def main(args):
    # initialize args from user

    # check num of args
    if len(args) > 6 or len(args) < 5:
        print("Invalid Input!")
        return

    # check k is a number
    k = args[1]
    try:
        k = int(args[1])
    except:
        print("Invalid Input!")
        return

    # check k > 1
    if k < 1:
        print("Invalid Input!")
        return

    max_iter = 300  # default

    if len(args) == 6:
        try:
            max_iter = int(args[2])
        except:
            print("Invalid Input!")
            return
        try:
            epsilon = float(args[3])
        except:
            print("Invalid Input!")
            return
        if epsilon < 0:
            print("Invalid Input!")
            return
        input_file_name1 = args[4]
        input_file_name2 = args[5]
    else:
        max_iter = 300
        try:
            epsilon = float(args[2])
        except:
            print("Invalid Input!")
            return
        
        input_file_name1 = args[3]
        input_file_name2 = args[4]

    # check max iter
    if max_iter < 1:
        print("Invalid Input!")
        return

    # check files names
    if (input_file_name1[-4:] != ".txt") and (input_file_name1[-4:] != ".csv"):
        print("Invalid Input!")
        return

    if (input_file_name2[-4:] != ".txt") and (input_file_name2[-4:] != ".csv"):
        print("Invalid Input!")
        return

    # reads input file into data points list
    data1 = pd.read_csv(input_file_name1, header=None)
    data2 = pd.read_csv(input_file_name2, header=None)
    data_merge = pd.merge(left=data1, right=data2, on=0, how='inner')
    data_merge = data_merge.sort_values(by=0, axis=0)
    data_points_indices = data_merge.iloc[:,0].tolist()
    for i in range(len(data_points_indices)):
        data_points_indices[i] = int(data_points_indices[i])
    data_points = data_merge.drop(columns=[0], axis=1).to_numpy()

    # check k<N
    if k >= len(data_points):
        print("Invalid Input!")
        return

    # returns centroids
    centroids, indices_arr = init_k_centroids(k, data_points,data_points_indices)
    for i in range(centroids.shape[0]):
        for j in range(centroids.shape[1]):
            centroids[i][j] = '%.4f' % centroids[i][j]
    centroids = pd.DataFrame(centroids, columns=[i for i in range(centroids.shape[1])]).to_numpy()

    # Output
    for i in range(len(indices_arr)):
        indices_arr[i] = str(int(indices_arr[i]))
    print(','.join(indices_arr))

    # centroids to array
    centroids_arr =[]
    vector_len = centroids.shape[1]
    for i in range(k):
        for j in range(vector_len):
                centroids_arr.append(centroids[i][j])
    
    # data_points to array
    data_points_arr =[]
    N = data_points.shape[0]
    for i in range(N):
        for j in range(vector_len):
                data_points_arr.append(data_points[i][j])

    # Interfacing with C extension
    # sending &k, &max_iter, &eps, &data_points_py, &centroids_py, &vector_len, &N
    k_means_centroids = mykmeanssp.fit(k, max_iter, epsilon, data_points_arr, centroids_arr, vector_len, N)
    for i in range(len(k_means_centroids)):
        k_means_centroids[i] = '%.4f' % k_means_centroids[i]

    for i in range(k):
        print(','.join(k_means_centroids[i*vector_len:(i+1)*vector_len]))

# MAIN
main(sys.argv)
