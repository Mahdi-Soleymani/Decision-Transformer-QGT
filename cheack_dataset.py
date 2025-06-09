import h5py
import numpy as np
from collections import defaultdict, Counter
import pickle

def analyze_dataset_by_upper_bounds(h5_file_path):
    result_dict = {}

    with h5py.File(f"{h5_file_path}.h5", 'r') as f:
        upper_bounds = f['upper_bounds'][:]
        mask_lengths = f['mask_lengths'][:]

        # Group mask_lengths by upper_bounds pattern
        pattern_dict = defaultdict(list)
        for ub, ml in zip(upper_bounds, mask_lengths):
            pattern_key = tuple(ub.tolist())
            pattern_dict[pattern_key].append(int(ml))

        # Find most common mask_length per pattern
        for pattern, lengths in pattern_dict.items():
            most_common_length = Counter(lengths).most_common(1)[0][0]
            result_dict[pattern] = most_common_length

    with open(f"{h5_file_path}.pkl", "wb") as f:
        pickle.dump(result_dict, f)

if __name__ == "__main__":
    result=analyze_dataset_by_upper_bounds("dataset_k4")
    with open("dataset_k7.pkl", "rb") as f:
        loaded_dict = pickle.load(f)
        print(len(loaded_dict))
    