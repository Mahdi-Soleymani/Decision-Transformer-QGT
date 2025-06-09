import numpy as np
import math

def generate_sorted_kronecker(k):
    # Base matrix: rows of [1,1] and [1,0]
    base = np.array([[1, 1],
                     [1, 0]], dtype=np.uint8)

    # Compute l = smallest power of 2 such that 2^l >= k
    l = 1
    while 2 ** l < k:
        l += 1

    # Kronecker product: repeat l times
    result = base
    for _ in range(l - 1):
        result = np.kron(result, base)

    # Each row is a vector; sort by number of 1s (descending)
    row_sums = result.sum(axis=1)
    sorted_indices = np.argsort(-row_sums)  # negative for descending
    sorted_result = result[sorted_indices]
    # Select k random columns (without replacement)
    total_cols = result.shape[1]
    chosen_indices = np.random.choice(total_cols, size=k, replace=False)
    selected_columns= result[:, chosen_indices]

    return selected_columns
    # return np.array([[1, 1, 1, 1, 1],
    #                  [1, 1, 1, 0, 0], 
    #                  [0, 0, 1, 1, 1],
    #                  [1, 0, 0, 0, 1],
    #                  [0, 1, 0, 1, 0]
    #                  ], dtype=np.uint8)
    # return np.array([[1, 1, 1],
    #                  [0, 1, 0], 
    #                  [0, 0, 1]
    #                  ], dtype=np.uint8)

# # Example usage
# k = 8
# mat = generate_sorted_kronecker(k)
# print(mat)
