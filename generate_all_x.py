from itertools import product
import numpy as np

def generate_all_x(k):
    """Generate all vectors x ∈ ℕᵏ such that sum(x) == k"""
    def helper(remaining, depth, current):
        if depth == 1:
            yield current + [remaining]
        else:
            for i in range(remaining + 1):
                yield from helper(remaining - i, depth - 1, current + [i])
    return list(helper(k, k, []))

def generate_all_x_half(x):
    """Generate all x_half where 0 ≤ x_half[i] ≤ x[i]"""
    return list(product(*[range(xi + 1) for xi in x]))

def generate_all_pairs(k):
    """Return list of (x, x_half) for all x ∈ ℕᵏ with sum k and x_half ≤ x"""
    pairs = []
    all_x = generate_all_x(k)
    for x in all_x:
        for x_half in generate_all_x_half(x):
            pairs.append((np.array(x, dtype=np.int8), np.array(x_half, dtype=np.int8)))
    return pairs


if __name__ == "__main__":
    pairs = generate_all_pairs(k=5)
    # for x, x_half in pairs:
    #     print("x:", x, "x_half:", x_half)
    print(len(pairs))

# from itertools import product
# import numpy as np

# def generate_all_pairs_lazy(k):
#     """Yield one (x, x_half) pair at a time.
    
#     x: non-negative int vector of length k summing to k
#     x_half: 0 ≤ x_half[i] ≤ x[i]
#     """
#     def generate_all_x():
#         """Generate all vectors x ∈ ℕᵏ such that sum(x) == k"""
#         def helper(remaining, depth, current):
#             if depth == 1:
#                 yield current + [remaining]
#             else:
#                 for i in range(remaining + 1):
#                     yield from helper(remaining - i, depth - 1, current + [i])
#         yield from helper(k, k, [])

#     for x in generate_all_x():
#         for x_half in product(*[range(xi + 1) for xi in x]):
#             yield np.array(x, dtype=np.int8), np.array(x_half, dtype=np.int8)
