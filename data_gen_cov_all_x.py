import numpy as np
import h5py
import random
from gurobipy import Model, GRB
from tqdm import tqdm
import argparse
import os
import multiprocessing as mp
import concurrent.futures
import math
import Xpair_gen as XP

def num_of_total_sols(k):
    """Calculates C(2k - 1, k) / k for a given k."""
    if k < 1: raise ValueError("k must be a positive integer.")
    return math.comb(2 * k - 1, k) 

def pad_sequence(seq, max_len, pad_value=0):
    seq = np.array(seq, dtype=np.float32)
    pad_size = max_len - seq.shape[0]
    if pad_size > 0:
        seq = np.pad(seq, (0, pad_size), mode='constant', constant_values=np.int8(pad_value))
    return seq

def pad_sequence2d(seq, max_len, pad_value=0):
    if len(seq) == 0:
        return np.full((max_len, 0), pad_value, dtype=np.int8)
    seq = [np.array(q, dtype=np.int8) for q in seq]
    num_sequences = len(seq)
    seq_length = len(seq[0])
    if num_sequences < max_len:
        pad_matrix = np.full((max_len - num_sequences, seq_length), int(pad_value), dtype=np.int8)
        seq = np.vstack((seq, pad_matrix))
    return seq

def generate_covariance_maximizing_sample(k, max_len, pad_scalar_val, pad_vec_val, x, x_half):
    # x = np.zeros(k, dtype=int)
    # x_half = np.zeros(k, dtype=int)
    # for i in range(k):
    #     idx = np.random.choice(k, 1)
    #     x[idx] += 1
    #     if random.random() < 0.5:
    #         x_half[idx] += 1

    x = x.reshape(-1, 1)
    x_half = x_half.reshape(-1, 1)

    model = Model("CovMax_ILP")
    model.setParam(GRB.Param.OutputFlag, 0)
    model.setParam(GRB.Param.Threads, 1)

    variables = [model.addVar(vtype=GRB.INTEGER, lb=0, ub=int(x[i].item()), name=f"x{i}") for i in range(k)]
    
    model.setParam(GRB.Param.PoolSearchMode, 2)
    max_num_of_sols_kept=num_of_total_sols(k)
    model.setParam(GRB.Param.PoolSolutions, max_num_of_sols_kept)
    
    model.setObjective(1, GRB.MAXIMIZE)
    model.optimize()
    num_solutions = model.SolCount
    q, r, rwrd = [], [], [np.log2(num_solutions)]
    num_of_constraints = 0
    is_solved = False
    while not is_solved:
        num_solutions = model.SolCount
        if num_solutions < 2:
            break
        solution_matrix = np.zeros((num_solutions, k))
        for sol_index in range(num_solutions):
            model.setParam(GRB.Param.SolutionNumber, sol_index)
            solution_matrix[sol_index] = [var.Xn for var in variables]
        cov_matrix = np.cov(solution_matrix, rowvar=False)

        model_cov = Model("Maximize_Variance")
        model_cov.setParam(GRB.Param.OutputFlag, 0)
        I = model_cov.addVars(k, vtype=GRB.BINARY, name="I")
        quad_expr = sum(I[i] * cov_matrix[i, j] * I[j] for i in range(k) for j in range(k))
        model_cov.setObjective(quad_expr, GRB.MAXIMIZE)
        model_cov.optimize()

        selected_indices = [i for i in range(k) if I[i].X > 0.5]
        selected_mask = np.zeros(k, dtype=int)
        for i in selected_indices:
            selected_mask[i] = 1
        q.append(selected_mask)
        new_result = np.matmul(selected_mask, x_half)[0]
        r.append(int(new_result))

        model.addConstr(sum(variables[i] for i in selected_indices) == new_result, name=f"c{num_of_constraints}")
        num_of_constraints += 1
        model.optimize()

        if model.status == GRB.OPTIMAL:
            if model.SolCount <= 1:
                is_solved = True
                rwrd.append(0)
            else:
                rwrd.append(np.log2(model.SolCount))
        else:
            rwrd.append(0)
            is_solved = True

    # rtg, s = [], 0
    # for reward in reversed(rwrd):
    #     s += reward
    #     rtg.append(s)
    # rtg = list(reversed(rtg))
    mask_length = min(len(rwrd), max_len)
    q_padded = pad_sequence2d(q[:max_len], max_len, pad_vec_val)
    r_padded = pad_sequence(r[:max_len], max_len, pad_scalar_val)
    rtg_padded = pad_sequence(rwrd[:max_len], max_len, pad_scalar_val)
    return q_padded, r_padded, rtg_padded, np.int8(mask_length), np.squeeze(x)

def generate_and_store_sample(worker_id, num_samples_per_worker, k, max_len, pad_scalar_val, pad_vec_val, file_prefix):
    file_name = f"{file_prefix}_{worker_id}.h5"
    sol_count=XP.count_x_xhalf_pairs(k)
    with h5py.File(file_name, 'w') as f:
        d_queries = f.create_dataset("queries", (num_samples_per_worker*sol_count, max_len, k), dtype='i1')
        d_results = f.create_dataset("results", (num_samples_per_worker*sol_count, max_len), dtype='i1')
        d_rtgs = f.create_dataset("rtgs", (num_samples_per_worker*sol_count, max_len), dtype='float')
        d_mask_lengths = f.create_dataset("mask_lengths", (num_samples_per_worker*sol_count,), dtype='i1')
        d_bounds = f.create_dataset("upper_bounds", (num_samples_per_worker*sol_count, k), dtype='i1')

        pbar = tqdm(total=num_samples_per_worker*sol_count, position=worker_id, desc=f"Worker {worker_id}", leave=True)
        generator = XP.XPairGenerator(k)
        sample_idx = 0
        while True:
            x, x_half, done = generator.get_next()
            if done:
                break
            for _ in range(num_samples_per_worker):
                q, r, rtg, mask_length, d_bound = generate_covariance_maximizing_sample(k, max_len, pad_scalar_val, pad_vec_val, x, x_half)
                d_queries[sample_idx] = q
                d_results[sample_idx] = r
                d_rtgs[sample_idx] = rtg
                d_mask_lengths[sample_idx] = mask_length
                d_bounds[sample_idx] = d_bound
                sample_idx += 1
                pbar.update(1)
        pbar.close()

    print(f"Worker {worker_id} saved {num_samples_per_worker} samples to {file_name}")

def merge_datasets(input_files, output_file):
    with h5py.File(output_file, 'w') as f_out:
        for i, file in enumerate(input_files):
            with h5py.File(file, 'r') as f_in:
                for key in f_in.keys():
                    data = f_in[key][:]
                    if key in f_out:
                        f_out[key].resize((f_out[key].shape[0] + data.shape[0]), axis=0)
                        f_out[key][-data.shape[0]:] = data
                    else:
                        maxshape = (None,) + data.shape[1:]
                        f_out.create_dataset(key, data=data, maxshape=maxshape, chunks=True)
    for file in input_files:
        os.remove(file)

def save_dataset_parallel(filename, num_samples, k, max_len, pad_scalar_val, pad_vec_val, num_cores):
    file_prefix = "temp_sample_file_worker"
    print(f"Using {num_cores} CPU cores for parallel processing...")
    samples_per_worker = num_samples // num_cores
    extra_samples = num_samples % num_cores

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for worker_id in range(num_cores):
            num_samples_for_worker = samples_per_worker + (1 if worker_id < extra_samples else 0)
            if num_samples_for_worker > 0:
                futures.append(executor.submit(
                    generate_and_store_sample,
                    worker_id,
                    num_samples_for_worker,
                    k,
                    max_len,
                    pad_scalar_val,
                    pad_vec_val,
                    file_prefix
                    
                ))
        concurrent.futures.wait(futures)

    worker_files = [f"{file_prefix}_{worker_id}.h5" for worker_id in range(num_cores) if os.path.exists(f"{file_prefix}_{worker_id}.h5")]
    merge_datasets(worker_files, filename)

def count_samples_in_h5(file_name):
    with h5py.File(file_name, 'r') as f:
        queries = f['queries']
        num_samples = queries.shape[0]
        print(f"Number of samples in {file_name}: {num_samples}")

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cores", type=int, default=6, help="Number of CPU cores to use")
    parser.add_argument('--num_samples', type=int, default=1, help='Total number of samples to generate')
    parser.add_argument('--file_name', type=str, default="dataset", help='Name of the output file')
    parser.add_argument("--k", type=int, default=3, help="Length of the query vector")

    args = parser.parse_args()
    k = args.k
    max_len = k
    pad_scalar_val = -10
    pad_vec_val = -30
    f_name = f"{args.file_name}_k{k}.h5"
    n_cores = min(args.n_cores, os.cpu_count())

    save_dataset_parallel(f_name, args.num_samples, k, max_len, pad_scalar_val, pad_vec_val, n_cores)
    count_samples_in_h5(f_name)

    file_size_mb = os.path.getsize(f_name) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")