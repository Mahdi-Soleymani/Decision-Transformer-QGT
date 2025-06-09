import h5py
import numpy as np
import argparse
import os

def filter_dataset_by_mask_length(input_path, output_path, desired_mask_length):
    with h5py.File(input_path, 'r') as fin:
        mask_lengths = fin['mask_lengths'][:]
        indices = np.where(mask_lengths == desired_mask_length)[0]
        print(f"Found {len(indices)} samples with mask_length = {desired_mask_length}")

        if len(indices) == 0:
            print("No matching samples found. Exiting.")
            return

        with h5py.File(output_path, 'w') as fout:
            for key in fin.keys():
                data = fin[key][indices]
                shape = (len(indices),) + data.shape[1:]
                fout.create_dataset(key, data=data, shape=shape, dtype=data.dtype)

    print(f"Filtered dataset saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--des_len", type=int, required=True, help="Desired mask length to filter by")
    parser.add_argument("--k", type=int, required=True, help="k")

    args = parser.parse_args()
    input_file = f"dataset_k{args.k}.h5"
    output_file=f"filtered_k{args.k}_len{args.des_len}.h5"
    filter_dataset_by_mask_length(input_file, output_file, args.des_len+1)
