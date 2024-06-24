## Overview
This project implements the K-means++ algorithm for initializing centroids for the K-means clustering algorithm. The implementation consists of both Python and C components. The Python component handles the K-means++ initialization, reading input files, and interfacing with the C extension. The C component is a C extension that performs the K-means clustering using the initial centroids provided by the Python component.

## Files
The project contains the following files:
1. `kmeans_pp.py`: Main interface of the program, written in Python.
2. `kmeans.c`: C extension containing the K-means algorithm.
3. `setup.py`: Setup file to build the C extension.

## Installation
**Build the C extension**: 
python3 setup.py build_ext --inplace

## Usage
### Main Program
To run the main program, use the following command:
python3 kmeans_pp.py <k> [<max_iter>] <eps> <file_name1> <file_name2>

Where:
- `k`: Number of required clusters.
- `max_iter` (Optional): Maximum number of K-means iterations (default is 300).
- `eps`: The epsilon value for convergence.
- `file_name1`: Path to the first input file (either `.txt` or `.csv`).
- `file_name2`: Path to the second input file (either `.txt` or `.csv`).

Example:
python3 kmeans_pp.py 3 100 0.01 input_1.txt input_2.txt

### Input Files
- The input files should contain N observations.
- The files can be either `.txt` or `.csv`.
- The files are combined using an inner join on the first column.

### Output
- The first line of the output contains the indices of the observations chosen as initial centroids.
- The subsequent lines contain the final centroids calculated by the K-means algorithm, with each centroid on a new line, formatted to 4 decimal places.

### Error Handling
- If invalid input is provided, the program prints "Invalid Input!" and terminates.
- If any other error occurs, the program prints "An Error Has Occurred" and terminates.

## Assumptions
1. Input files are in the correct format.
2. Command-line arguments are in the correct format.
3. All data points are unique.
4. Uses double in C and float in Python for vector elements.
```

Submit the zip file via Moodle.

---

Thank you for using this K-means++ and K-means implementation!
