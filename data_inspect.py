import h5py

file_path = "dataset_k3.h5"  # or f"{f_name}.h5"
#file_path = "data_cov/k_9/10M_k9.h5"  # or f"{f_name}.h5"


with h5py.File(file_path, "r") as f:
    print("Keys in dataset:", list(f.keys()))  # List all datasets/groups
    
    # Example: print first few entries from each dataset
    for key in f.keys():
        data = f[key][:]
        print(f"\nDataset '{key}' shape: {data.shape}")
        print(f"First few entries of '{key}':\n", data[30:40])
        if key=="rtgs":
            print(f"average", data[:,0].mean())
