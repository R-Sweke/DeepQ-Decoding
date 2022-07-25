# ------------ This script generates the initial grid over which we will start our search ----------------

import os
import shutil
import pickle
cwd = os.getcwd()

# ------------ the fixed parameters: These are constant for all error rates -----------------------------

fixed_config = {"d": 5,
                "use_Y": False,
                "train_freq": 1,
                "batch_size": 32,
                "print_freq": 250,
                "rolling_average_length": 1000,
                "stopping_patience": 1000,
                "error_model": "X",
                "c_layers": [[64,3,2],[32,2,1],[32,2,1]],
                "ff_layers": [[512,0.2]],
                "max_timesteps": 1000000,
                "volume_depth": 5,
                "testing_length": 101,
                "buffer_size": 50000,
                "dueling": True,
                "masked_greedy": False,
                "static_decoder": True}

fixed_config_path = os.path.join(cwd, "../fixed_config.p")
pickle.dump(fixed_config, open(fixed_config_path, "wb" ) )

# ---------- The variable parameters grid --------------------------------------------------------------

p_phys = 0.001
success_threshold = 100000

learning_starts_list = [1000]
learning_rate_list = [0.0001, 0.00005, 0.00001]
exploration_fraction_list = [100000, 200000]
sim_time_per_ef = [10, 10]
max_eps_list = [1.0]
target_network_update_freq_list = [2500, 5000]
gamma_list = [0.99]
final_eps_list = [0.04, 0.02, 0.001]

config_counter = 1
for ls in learning_starts_list:
    for lr in learning_rate_list:
        for ef_count, ef in enumerate(exploration_fraction_list):
            for me in max_eps_list:
                for tnuf in target_network_update_freq_list:
                    for g in gamma_list:
                        for fe in final_eps_list:

                            variable_config_dict = {"p_phys": p_phys,
                            "p_meas": p_phys,
                            "success_threshold": success_threshold,
                            "learning_starts": ls,
                            "learning_rate": lr,
                            "exploration_fraction": ef,
                            "max_eps": me,
                            "target_network_update_freq": tnuf,
                            "gamma": g,
                            "final_eps": fe}

                            config_directory = os.path.join(cwd,"config_"+str(config_counter)+"/")
                            if not os.path.exists(config_directory):
                                os.makedirs(config_directory)
                            else:
                                shutil.rmtree(config_directory)           #removes all the subdirectories!
                                os.makedirs(config_directory)

                            file_path = os.path.join(config_directory, "variable_config_"+str(config_counter) + ".p")
                            pickle.dump(variable_config_dict, open(file_path, "wb" ) )
                                        
                            # Now, write into the bash script exactly what we want to appear there
                            job_limit = str(sim_time_per_ef[ef_count])
                            job_name=str(p_phys)+"_"+str(config_counter)
                            output_file = os.path.join(cwd,"output_files/out_"+job_name+".out")
                            error_file = os.path.join(cwd,"output_files/err_"+job_name+".err")
                            python_script = os.path.join(cwd, "Single_Point_Training_Script.py")


                            f = open(config_directory + "/simulation_script.sh",'w')  
                            f.write('''#!/bin/bash

#SBATCH --job-name='''+job_name+'''          # Job name, will show up in squeue output
#SBATCH --ntasks=4                           # Number of cores
#SBATCH --nodes=1                            # Ensure that all cores are on one machine
#SBATCH --time=0-'''+job_limit+''':30:00    # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=1000                   # Memory per cpu in MB (see also --mem) 
#SBATCH --output='''+output_file+'''         # File to which standard out will be written
#SBATCH --error='''+error_file+'''           # File to which standard err will be written

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID


# ---------------------- JOB SCRIPT ---------------------------------------------

# ----------- Activate the environment  -----------------------------------------

module load miniconda
source activate deepq-mkl

# ----------- Tensorflow XLA flag -----------------------------------------------
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit

# ------- run the script -----------------------

python -u'''+python_script+''' '''+str(config_counter)+''' || exit 1

#----------- wait some time ------------------------------------

sleep 50''')
                            f.close()
                            config_counter += 1 
