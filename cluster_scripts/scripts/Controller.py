import os 
import json
import pickle
import subprocess
import shutil
from shutil import copyfile, rmtree
import datetime

# --- Helper function to clean redundant files -------------------------------------------------------------------------------------

def clean_config_dir(p_phys, keep_num):
    """
    In order to free up space, we have to delete config directories during training.
    This function should be called when training at one physical error rate is completed.
    The function reads the `/results/results_from_x.txt' file and ranks configs according
    to average qubit lifetime. It then deletes all but best `keep_num` configs.

    Params
    ------
    p_phys: Physical error rate for which we want to delete config directories
    keep_num: Number of config directories to keep
    """
    result_file = os.path.join(os.getcwd(),f"results/results_from_{p_phys}.txt")
    ranking = {}
    with open(result_file) as f:
            lines = f.readlines()
    num_del = max(len(lines) - keep_num, 0)
    for line in lines:
            s = line.strip().split(": ")
            ranking[s[0]] = float(s[1])

    ranking = dict(sorted(ranking.items(), key=lambda item: item[1]))

    for rank, key in enumerate(ranking):
            if rank >= int(num_del):
                    break
            # delete config directory
            dir_to_delete = os.path.join(os.getcwd(),f"{p_phys}/config_{key}")
            print(f"Deleting : {dir_to_delete}")
            rmtree(dir_to_delete)

# --- First we set the controller parameters ---------------------------------------------------------------------------------------

cwd= os.getcwd()
now = datetime.datetime.now()
print(now)                          # This allows us to track executions in stdout

with open('../training_config.json', 'r') as f:
    data = json.load(f)
    parameter_grid = data["controller_params"]["controller_param_grid"]
    controller_params = data["controller_params"]

# --- Then we figure out the current error rate of running simulations ----------------------------------------------------------------

file_path_to_error_rate = os.path.join(cwd,"current_error_rate.txt")
with open(file_path_to_error_rate) as f:
    content = f.readlines()
if "\n" in content[0]:
    new_line_index = content[0].index("\n")
    current_error_rate = content[0][:new_line_index]
else:
    current_error_rate = content[0]

# ---- Now we calculate how many simulations ran for this error rate ------------------------------------------------------------------

configs = []
config_counter = 1
check_directory = os.path.join(cwd, current_error_rate+str("/"))
for directory in os.walk(check_directory):
    if "config" in directory[0]:
        configs.append(directory[0].split("config_")[1])
        config_counter +=1
        
num_configs = config_counter - 1

# ---- Now we fetch all the currently available results for this error rate -----------------------------------------------------------

completed_simulations = 0
results_dict = {}
for config in configs:
    folder = os.path.join(check_directory,"config_"+str(config)+"/")
    available_files = os.listdir(folder)
    if 'results.p' in available_files:
        # If the results file is available then we add the result, and mark the simulation as being completed
        results = pickle.load(open(folder+"results.p", "rb" ))
        results_dict[str(config)] = results[-1:][0]
        completed_simulations +=1
    else:
        # If testing didn't pass current error rate, the training will get stuck. Therefore we check if 'all_results.p' is available.
        if "all_results.p" in available_files:
            results_dict[str(config)] = 0
            completed_simulations +=1
        elif "started_at.p" in available_files:
            # if we know that the simulation started, then we see how long it has been running for
            sim_start_time = pickle.load(open(folder+"started_at.p", "rb" ))
            time_diff = now - sim_start_time
            time_diff_hours = time_diff.total_seconds()/(3600.0)
            if time_diff_hours > controller_params["simulation_time_limit_hours"]:
                results_dict[str(config)] = 0
                completed_simulations +=1
            else:
                results_dict[str(config)] = "still running"
        else:
            results_dict[str(config)] = "not started"

# ---- Here we write out the results to keep track of what is going on ----------------------------------------------------------------

has_written = False
for config in configs:
    results_path = os.path.join(cwd,"results/results_from_"+current_error_rate+".txt")
    if not has_written:
        text_file = open(results_path, "w")
        text_file.write(str(config)+": "+str(results_dict[str(config)])+"\n")
        text_file.close()
        has_written = True
    else:
        text_file = open(results_path, "a")
        text_file.write(str(config)+": "+str(results_dict[str(config)])+"\n")
        text_file.close()

# ---- We also write to the history file -----------------------------------------------------------------------------------------------

history_path = os.path.join(cwd,"history.txt")
text_file = open(history_path, "a")
text_file.write("""--------- """+now.strftime("%Y-%m-%d %H:%M")+""" --------------

current error rate: """+current_error_rate+"""
total simulations: """+str(num_configs)+"""
finished simulations: """+str(completed_simulations)+"""

""")
text_file.close()


# ---- Now we check if all simulations for current error rate are finished and spawn new ones or end the process  ---------------------


# We check if all the simulations are finished
if completed_simulations == num_configs:

    # We calculate which results are eligible by virtue of passing the threshold
    threshold = controller_params["threshold_dict"][current_error_rate]
    eligible_results = []
    for key in results_dict.keys():
        if not isinstance(results_dict[key], str):
            if results_dict[key] > threshold:
                eligible_results.append(results_dict[key])
            
    # If there are eligible results then we need to spawn new simulations
    if len(eligible_results) > 0:
        # First we sort the eligible results and find the best ones
        sorted_results = sorted(eligible_results, reverse=True) 
        top_results = sorted_results[:controller_params["num_best_to_spawn_from"]]


        # Then we find and write out the configurations corresponding to the best results
        best_results_path = os.path.join(cwd,"results/best_results_from_"+current_error_rate+".txt")
        top_counter = 0
        has_written = False
        top_configurations = {}
        for result in top_results:
                for key in results_dict.keys():
                    if not isinstance(results_dict[key], str):
                        if results_dict[key] == result and key not in top_configurations.keys():
                            if top_counter < controller_params["num_best_to_spawn_from"]:
                                top_configurations[key] = results_dict[key]
                                top_counter +=1 
                                if not has_written:
                                    text_file = open(best_results_path, "w")
                                    text_file.write(key+": "+str(top_configurations[key])+"\n")
                                    text_file.close()
                                    has_written = True
                                else:
                                    text_file = open(best_results_path, "a")
                                    text_file.write(key+": "+str(top_configurations[key])+"\n")
                                    text_file.close()
                            else:
                                break 
               
            
        # Now we need to either spawn new simulations, or end the process:

        old_p_phys_index = controller_params["p_phys_list"].index(float(current_error_rate))
        new_p_phys_index = old_p_phys_index + 1

        if new_p_phys_index == len(controller_params["p_phys_list"]):
            # In this case, we have reached the end, and the process should end without spawning new simulations

            text_file = open(history_path, "a")
            text_file.write("All error rates have been completed - simulations are done!\n\n")
            text_file.close()

        else:
            # In this case we need to spawn all the new simulations. In particular:
            # For each spawning network we need to:
            #    - generate a new grid, with each config having a new folder
            #    - copy the base DQN into the relevant folder
            #    - create the new variable configs in that folder
            #    - create a simulation script in that folder


            new_p_phys = controller_params["p_phys_list"][new_p_phys_index]
            new_p_phys_directory = os.path.join(cwd,str(new_p_phys)+"/")

            text_file = open(history_path, "a")
            text_file.write("Spawning new simulations from "+str(top_counter)+" simulations which surpassed the threshold.\n\n")
            text_file.close()
            
            config_counter = 1
            # We go through each spawning network
            for spawn in top_configurations.keys():
                
                # We create a grid from this network
                for ls in parameter_grid["learning_starts_list"]:
                    for lr in parameter_grid["learning_rate_list"]:
                        for ef in parameter_grid["exploration_fraction_list"]:
                            for me in parameter_grid["max_eps_list"]:
                                for tnuf in parameter_grid["target_network_update_freq_list"]:
                                    for g in parameter_grid["gamma_list"]:
                                        for fe in parameter_grid["final_eps_list"]:
                                        
                                            # We create the folder and put the configs script inside
                                            variable_config_dict = {"p_phys": new_p_phys,
                                                "p_meas": new_p_phys,
                                                "success_threshold": controller_params["success_threshold_list"][new_p_phys_index],
                                                "learning_starts": ls,
                                                "learning_rate": lr,
                                                "exploration_fraction": ef,
                                                "max_eps": me,
                                                "target_network_update_freq": tnuf,
                                                "gamma": g,
                                                "final_eps": fe}

                                            config_directory = os.path.join(new_p_phys_directory, "config_"+str(config_counter))
                                            if not os.path.exists(config_directory):
                                                os.makedirs(config_directory)
                                            else:
                                                shutil.rmtree(config_directory)           #removes all the subdirectories!
                                                os.makedirs(config_directory)

                                            new_config_path = os.path.join(config_directory, "variable_config_"+str(config_counter) + ".p")
                                            pickle.dump(variable_config_dict, open(new_config_path, "wb" ) )

                                            new_sim_script_path = os.path.join(config_directory, "simulation_script.sh")

                                            # Now, write into the bash script exactly what we want to appear there
                                            python_script = os.path.join(new_p_phys_directory,"Single_Point_Continue_Training_Script.py")
                                            job_name=str(new_p_phys)+"_"+str(config_counter)
                                            output_file = os.path.join(new_p_phys_directory,"output_files/out_"+job_name+".out")
                                            error_file = os.path.join(new_p_phys_directory,"output_files/err_"+job_name+".err")

                                            f = open(new_sim_script_path,"w")                  
                                            f.write('''#!/bin/bash

#SBATCH --job-name={job_name}                # Job name, will show up in squeue output
#SBATCH --ntasks=4                           # Number of cores
#SBATCH --nodes=1                            # Ensure that all cores are on one machine
#SBATCH --time=0-15:30:00                    # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=1000                   # Memory per cpu in MB (see also --mem) 
#SBATCH --output={output_file}               # File to which standard out will be written
#SBATCH --error={error_file}                 # File to which standard err will be written

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID


# ---------------------- JOB SCRIPT ---------------------------------------------

# ----------- Activate the environment  -----------------------------------------

module load miniconda
source activate deepq-mkl

# ----------- Tensorflow XLA flag -----------------------------------------------

export TF_XLA_FLAGS=--tf_xla_cpu_global_jit

# ------- run the script -----------------------

python {python_script} {config_counter} {new_p_phys_directory} || exit 1'''.format(job_name=job_name,
                                                                           output_file=output_file,
                                                                           error_file=error_file,
                                                                           python_script=python_script,
                                                                           config_counter=config_counter,
                                                                           new_p_phys_directory=new_p_phys_directory))
                                            f.close()
                                            
                                            # Finally I copy the base neural network that will be loaded into that folder
                                            source_weights = os.path.join(check_directory,"config_"+spawn+"/final_dqn_weights.h5f")
                                            destination_weights = os.path.join(config_directory,"initial_dqn_weights.h5f")
                                            copyfile(source_weights, destination_weights)

                                            source_memory = os.path.join(check_directory,"config_"+spawn+"/memory.p")
                                            destination_memory = os.path.join(config_directory,"memory.p")
                                            copyfile(source_memory, destination_memory)

                                            config_counter += 1 
                                        
            # Now, I want to run all the simulation scripts that I have just generated...
            path_to_sims_script = os.path.join(new_p_phys_directory, "Start_Continuing_Simulations.sh")
            subprocess.call([path_to_sims_script, new_p_phys_directory])                                   # NB! This has to be executable!!

            # Now clean configs
            clean_config_dir(current_error_rate, 5)

            # Finally we update the current error rate text file
            text_file = open(file_path_to_error_rate, "w")
            text_file.write(str(new_p_phys))
            text_file.close()

    else:
        text_file = open(history_path, "a")
        text_file.write("All simulations finished, but none surpassed the threshold. Training stops here. \n\n")
        text_file.close()
else:
    text_file = open(history_path, "a")
    text_file.write("Will continue waiting for all simulations to finish. \n\n")
    text_file.close()

        

