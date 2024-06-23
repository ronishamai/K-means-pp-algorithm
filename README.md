#### Overview

This project involves implementing the K-means++ algorithm in Python and integrating it with a C extension that implements the core K-means clustering algorithm. The goal is to read data from two input files, apply K-means++ for centroid initialization, and then utilize the C extension for the actual clustering process. Here's a breakdown of the main components and how they were implemented.

#### Files in the Project

1. **kmeans_pp.py**
   - **Role**: Acts as the main interface of the program.
   - **Functionality**:
     - Parses command-line arguments including `k` (number of clusters), `max_iter` (maximum iterations for K-means), `eps` (convergence threshold), `file_name1`, and `file_name2` (paths to input data files).
     - Reads and merges data from `file_name1` and `file_name2` using a common key.
     - Implements the K-means++ algorithm using NumPy for centroid initialization, ensuring reproducibility with `np.random.seed(0)`.
     - Interfaces with the C extension (`mykmeanssp`) by passing initial centroids and data for clustering.
     - Prints the indices of initial centroids chosen by K-means++ and the final centroids after clustering, formatted to 4 decimal places.

2. **kmeans.c**
   - **Role**: C extension module implementing the K-means algorithm.
   - **Functionality**:
     - Named `mykmeanssp`.
     - Provides a `fit()` function that receives initial centroids and data points.
     - Executes the K-means clustering algorithm excluding the step for centroid initialization (handled by K-means++ in Python).
     - Returns final centroids calculated after convergence.

3. **setup.py**
   - **Role**: Setup script for building the C extension.
   - **Functionality**:
     - Configures the build process to compile the C extension module `mykmeanssp`.
     - Executes cleanly without errors or warnings using `python3 setup.py build_ext --inplace`.

#### Assumptions and Considerations

- **Input Format**: Assumes correct formatting of input files (`txt` or `csv`).
- **Error Handling**: Handles invalid inputs by printing appropriate error messages and terminating gracefully.
- **Documentation**: Code is well-commented for clarity and includes citations for any borrowed code.

#### Build and Execution

- **Building**: Use `python3 setup.py build_ext --inplace` to build the C extension.
- **Execution**: Run `kmeans_pp.py` with specified command-line arguments to perform K-means++ initialization, interface with C extension for clustering, and output results as specified.
