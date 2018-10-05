#!/bin/bash

#SBATCH --job-name=0.001_24          # Job name, will show up in squeue output
#SBATCH --ntasks=4                           # Number of cores
#SBATCH --nodes=1                            # Ensure that all cores are on one machine
#SBATCH --time=0-14:30:00    # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=1000                   # Memory per cpu in MB (see also --mem) 
#SBATCH --output=/bee/z00/scratch/danilit/scripts/2018_10_02/d5_dp_fm_v3/0.001/output_files/out_0.001_24.out         # File to which standard out will be written
#SBATCH --error=/bee/z00/scratch/danilit/scripts/2018_10_02/d5_dp_fm_v3/0.001/output_files/err_0.001_24.err           # File to which standard err will be written

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID


# ---------------------- JOB SCRIPT ---------------------------------------------

# ----------- Activate the environment  -----------------------------------------

#module load python/3.6.5

# ------- run the script -----------------------

python /bee/z00/scratch/danilit/scripts/2018_10_02/d5_dp_fm_v3/0.001/Single_Point_Training_Script.py 24

#----------- wait some time ------------------------------------

sleep 50