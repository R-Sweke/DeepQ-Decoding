import os 
import pickle
import subprocess
import shutil
from shutil import copyfile
import datetime

# --- First we set the controller parameters ---------------------------------------------------------------------------------------

cwd= os.getcwd()
now = datetime.datetime.now()
print(now)                          # This allows us to track executions in stdout

# These are TESTING thresholds to determine which simulations should be spawned off of - should be chosen wrt to some benchmark - i.e. mwpm!
threshold_dict = {"0.001": 1000, "0.002": 500, "0.003": 334, "0.004": 250, "0.005": 200, "0.006": 167, "0.007": 142, "0.008": 125, "0.009": 112, "0.01": 100}    
num_best_to_spawn_from = 1

# These are TRAINING thresholds to be used when an individual script should converge
p_phys_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
success_threshold_list = [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000]             

# This is the amount of time we give to each simulation before marking it as timed out
simulation_time_limit_hours = 12.1              

# Grid over which any spawned simulation will run:

learning_starts_list = [1000]
learning_rate_list = [0.0005, 0.0001, 0.00005, 0.00001]
exploration_fraction_list = [100000, 200000, 300000]
max_eps_list = [1.0, 0.5, 0.25]
target_network_update_freq_list = [2500, 5000]
gamma_list = [0.99]
final_eps_list = [0.04, 0.02, 0.001]

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
        configs.append(config_counter)
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
        if "started_at.p" in available_files:
            # if we know that the simulation started, then we see how long it has been running for
            sim_start_time = pickle.load(open(folder+"started_at.p", "rb" ))
            time_diff = now - sim_start_time
            time_diff_hours = time_diff.seconds/(3600.0)
            if time_diff_hours > simulation_time_limit_hours:
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
    threshold = threshold_dict[current_error_rate]
    eligible_results = []
    for key in results_dict.keys():
        if not isinstance(results_dict[key], str):
            if results_dict[key] > threshold:
                eligible_results.append(results_dict[key])
            
    # If there are eligible results then we need to spawn new simulations
    if len(eligible_results) > 0:
        # First we sort the eligible results and find the best ones
        sorted_results = sorted(eligible_results, reverse=True) 
        top_results = sorted_results[:num_best_to_spawn_from]


        # Then we find and write out the configurations corresponding to the best results
        best_results_path = os.path.join(cwd,"results/best_results_from_"+current_error_rate+".txt")
        top_counter = 0
        has_written = False
        top_configurations = {}
        for result in top_results:
                for key in results_dict.keys():
                    if not isinstance(results_dict[key], str):
                        if results_dict[key] == result and key not in top_configurations.keys():
                            if top_counter < num_best_to_spawn_from:
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

        old_p_phys_index = p_phys_list.index(float(current_error_rate))
        new_p_phys_index = old_p_phys_index + 1

        if new_p_phys_index == len(p_phys_list):
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


            new_p_phys = p_phys_list[new_p_phys_index]
            new_p_phys_directory = os.path.join(cwd,str(new_p_phys)+"/")

            text_file = open(history_path, "a")
            text_file.write("Spawning new simulations from "+str(top_counter)+" simulations which surpassed the threshold.\n\n")
            text_file.close()
            
            config_counter = 1
            # We go through each spawning network
            for spawn in top_configurations.keys():
                
                # We create a grid from this network
                for ls in learning_starts_list:
                    for lr in learning_rate_list:
                        for ef in exploration_fraction_list:
                            for me in max_eps_list:
                                for tnuf in target_network_update_freq_list:
                                    for g in gamma_list:
                                        for fe in final_eps_list:
                                        
                                            # We create the folder and put the configs script inside
                                            variable_config_dict = {"p_phys": new_p_phys,
                                                "p_meas": new_p_phys,
                                                "success_threshold": success_threshold_list[new_p_phys_index],
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
#SBATCH --time=0-10:00:00                    # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=1000                   # Memory per cpu in MB (see also --mem) 
#SBATCH --output={output_file}               # File to which standard out will be written
#SBATCH --error={error_file}                 # File to which standard err will be written
#SBATCH --mail-type=ALL                      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=rsweke@gmail.com         # Email to which notifications will be sent 

# store job info in output file, if you want...
scontrol show job $SLURM_JOBID


# ---------------------- JOB SCRIPT ---------------------------------------------

# ----------- Activate the environment  -----------------------------------------

module load python/3.6.5

# ------- run the script -----------------------

python {python_script} {config_counter} {new_p_phys_directory}'''.format(job_name=job_name,
                                                                           output_file=output_file,
                                                                           error_file=error_file,
                                                                           python_script=python_script,
                                                                           config_counter=config_counter,
                                                                           new_p_phys_directory=new_p_phys_directory))
                                            f.close()
                                            
                                            # Finally I copy the base neural network that will be loaded into that folder
                                            source_weights = os.path.join(check_directory,"config_"+spawn+"/dqn_weights.h5f")
                                            destination_weights = os.path.join(config_directory,"dqn_weights.h5f")
                                            copyfile(source_weights, destination_weights)

                                            source_memory = os.path.join(check_directory,"config_"+spawn+"/memory.p")
                                            destination_memory = os.path.join(config_directory,"memory.p")
                                            copyfile(source_memory, destination_memory)

                                            config_counter += 1 
                                        
            # Now, I want to run all the simulation scripts that I have just generated...
            path_to_sims_script = os.path.join(new_p_phys_directory, "Start_Continuing_Simulations.sh")
            subprocess.call([path_to_sims_script, new_p_phys_directory])                                   # NB! This has to be executable!!

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

        

