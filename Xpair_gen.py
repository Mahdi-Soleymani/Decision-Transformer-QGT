import numpy as np
from itertools import product

class XPairGenerator:
    def __init__(self, k):
        self.k = k
        self._x_list = list(self._generate_all_x(k))
        self._current_x_index = 0
        self._current_x_half_iter = None

    def _generate_all_x(self, total, length=None):
        if length is None:
            length = total
        def helper(remaining, depth, current):
            if depth == 1:
                yield current + [remaining]
            else:
                for i in range(remaining + 1):
                    yield from helper(remaining - i, depth - 1, current + [i])
        yield from helper(total, length, [])

    def get_next(self):
        while self._current_x_index < len(self._x_list):
            x = self._x_list[self._current_x_index]
            if self._current_x_half_iter is None:
                self._current_x_half_iter = product(*[range(xi + 1) for xi in x])
            try:
                x_half = next(self._current_x_half_iter)
                return np.array(x, dtype=np.int8), np.array(x_half, dtype=np.int8), False
            except StopIteration:
                self._current_x_index += 1
                self._current_x_half_iter = None
        return None, None, True  # All combinations exhausted

# # Example usage:
# generator = XPairGenerator(k=5)
# results = []
# while True:
#     x, x_half, done = generator.get_next()
#     print((x, x_half, done))
#     if done:
#         break
#     results.append((x, x_half))

import itertools
import math

def count_x_xhalf_pairs(k):
    def compositions(n, k):
        """Generate all integer compositions of n into k non-negative integers."""
        if k == 1:
            yield (n,)
        else:
            for i in range(n + 1):
                for tail in compositions(n - i, k - 1):
                    yield (i,) + tail

    total = 0
    for x in compositions(k, k):
        num_x_half = 1
        for xi in x:
            num_x_half *= (xi + 1)  # choices from 0 to xi inclusive
        total += num_x_half
    return total

#print(count_x_xhalf_pairs(5))



