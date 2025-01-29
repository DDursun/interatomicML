#!/bin/bash

#SBATCH --job-name=Be
#SBATCH --nodes=1 # Run on 2 node
#SBATCH --exclusive # this job takes all the node's CPUs
#SBATCH --partition=amd  # SLURM partition (amd or intel)
#SBATCH --time=48:00:00 # Time limit hrs:min:sec. If not specified, time limit is infinity
#SBATCH --output=output_50.log # Standard output and error log

mpirun -n 48 python qACE.py
