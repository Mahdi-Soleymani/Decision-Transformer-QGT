#!/bin/bash

# Define the directory containing the logs and output files
LOG_DIR="./"  # Adjust this path if your logs are in another directory

# Remove output and error log files (e.g., files matching "job.*.out" or "output.*.log")
echo "Cleaning up output and error logs..."

rm -f ${LOG_DIR}*.log
rm -f ${LOG_DIR}job.*.out
rm -f ${LOG_DIR}output_lr.*.log
rm -f ${LOG_DIR}slurm*.out  # If there are SLURM-specific logs (e.g., slurm-<jobid>.out)
rm -f ${LOG_DIR}*.out
rm -f ${LOG_DIR}*.err
rm -f *.pth
# You can also specify other file patterns as needed (e.g., temporary files)

echo "Cleanup completed!"
