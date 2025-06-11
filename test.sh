#!/bin/bash

LOG_FILE="test_all_dlen.log"
mkdir -p logs
echo "Starting test sweep..." > "$LOG_FILE"

for i in {1..7}; do
    echo "Running with --des_len=$i" | tee -a "$LOG_FILE"

    python atari/test_QGT.py \
        --des_len $i \
        --num_iter 1000 \
        --num_cores 6 \
        --k 7 \
        --checkpoint_cov   stoic-jazz-2.pth \
        --checkpoint_rand  stoic-jazz-2.pth \
        --mode DT \
        --pickle models/dataset_k7.pkl
        >> "$LOG_FILE" 2>&1
done

echo "All tests completed!" | tee -a "$LOG_FILE"
