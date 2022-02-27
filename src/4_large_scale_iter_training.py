#!/usr/bin/env python
# coding: utf-8

# #### 4) Large Scale Iterative Training and Hyper-Parameter Optimization
# 
# Now that we have seen how to train and test decoders at a fixed error rate, for a given set of hyper-parameters, we would like to turn our attention to how we might be able to obtain good decoders for a large range of error rates. In order to achieve this we have developed an iterative training procedure involving hyper-parameter searches at each level of the iteration. In this notebook we will first outline the procedure before proceeding to discuss in detail how one can implement this procedure on a high-performance computing cluster. The scripts required for this implementation are contained in the cluster_scripts folder of the repo.
# 
# ##### 4a) Outline of the Procedure
# 
# As illustrated in the figure below, the fundemental idea is to iterate through increasing error rates, performing hyper-parameter optimizations at each iteration, and using various attributes of the optimization at one step of the iteration as a starting point for the subsequent step.

# <p align="center">
# <img src="Images/grid_search_procedure.png" width="70%" height="70%">
# </p>

# In particular, the procedure works as follows:
# 
#   1. We begin by fixing the error rate to some small initial value which we estimate to be far below the threshold of the decoder and at which the agent will be able to learn a good strategy.
#   2. We then set the fixed hyper-parameters (as discussed in the previous section) which will remain constant throughout the entire training procedure, even as we increment the error rate. 
#   3. Next, we create a hyper-parameter grid over the hyperparameters which we would like to optimize (the variable hyper-parameters) and proceed to train a decoder for each set of hyper-parameters in the grid. For each of these initial simulations the decoder/agent is initialized with no memory and with random initial weights.
#   4. For each point in the hyperparameter grid we store 
#       - The hyper-parameter settings for this point.
#       - The entire training history.
#       - The weights of the final agent.
#       - The memory of the agent.
#   5. Additionally, for each trained agent we then evaluate the agent in "test-mode" and record the results (i.e. the average qubit lifetime one can expect when using this decoder in practice).
#   6. Now, given all "test-mode" results for all obtained decoders at this error rate, we then filter the results and identify the best performing decoder.
#   7. At this point, provided the results from just completed iteration are above some specified performance threshold, we then increase the error rate. To do this we start by fetching the experience memory and weights of the optimal decoder from the just completed iteration. Then, at the increased error rate we create a new hyper-parameter grid over the variable hyper-parameters, and train a decoder at each point in this new hyper-parameter grid. However, in this subsequent step of the iteration the agents are not initialized with random memories and weights, but with the memory and weights of the optimal performing decoder from the previous iteration.
#   8. This procedure then iterates until the point at which, with respect to the current error rate, the logical qubit lifetime when corrected by the optimal decoder falls beneath that of a single faulty-qubit - i.e until we are able to identify the pseudo-threshold of the decoder.
#       

# ##### 4a) Practically Implementing Large Scale Iterative Training on an HPC Cluster

# In the cluster_scripts folder of the repo we provide scripts for practically implementing the above iterating training procedure on an HPC cluster using the slurm workload manager. In this section we provide detailed documentation for how to set up and implement this procedure using the provided scripts.
# 
# As an example, we will work through the iterative training procedure in detail for d=5 and X noise only, although the steps here can be easily modified to different physical scenarios, and we have also provided all the scripts necessary for d=5 with depolarizing noise. 
# 
# Before beginning make sure that base "../d5_x/" directory contains:
# 
#    - Environments.py
#    - Function_Library.py
#    - Controller.py
#    - make_executable.sh
#    - static_decoder (an appropriate referee decoder with the corresponding lattice size and error model)
#    - An empty folder called "results" 
#    - An empty text document called "history.txt"
#    - A subdirectory for each error rate one would like to iterate through
#    - A text file "current_error_rate.txt" containing one line with the lowest error rate - i.e. 0.001
#     
# Furthermore, the subdirectory corresponding to the lowest error rate (0.001 here) should contain:
# 
#    - Environments.py
#    - Function_Library.py
#    - Generate_Base_Configs_and_Simulation_Scripts.py
#    - Single_Point_Training_Script.py
#    - Start_Simulations.sh
#    - An empty folder "output_files"
#     
# Additionally every other error rate subdirectory should contain:
# 
#    - Environments.py
#    - Function_Library.py
#    - Single_Point_Continue_Training_Script.py
#    - Start_Continuing_Simulations.sh
#    - An empty folder "output_files"
# 
# 
# In order to run a customized/modified version of this procedure this exact directory and file structure should be replicated, as we will see below all that is necessary is to modify:
# 
#    - Controller.py
#    - Generate_Base_Configs_and_Simulation_Scripts.py
#    - Providing the appropriate static/referee decoder
#    - Renaming the error rate subdirectories appropriately
# 
# To begin, copy the entire folder "d5_x" onto the HPC cluster and navigate into the "../d5_x/" directory. Then:
# 
# 1) From "../d5_x/" start a "screen" - this provides a persistent terminal which we can later detach and re-attach at will to keep track of the training procedure, without having to remain logged in to the cluster.
# 
#      a) type "screen"
#      b) then press enter
#      c) We are now in a screen
#      
# 2) Run the command "bash make_executable.sh". This will allow the controller - a python script which will be run periodically to control the training process - to submit jobs via slurm.
# 
# 3) Using vim or some other in-terminal editor, modify the following in Controller.py:
#     
#     a) Set all the error rates that you would like to iterate through - make sure there is an appropriate subdirectory for each error rate given here.
#     b) For each error rate, provide the expected lifetime of a single faulty qubit (i.e. the threshold for decoding sucess) as well as the average qubit lifetime you would like to use as a threshold for stopping training. We recommend setting this training threshold extremely high, so that training ends due to convergence.
#     c) set the hyper-parameter grid that you would like to use at each error rate iteration.
#     d) Also make sure that all the cluster parameters (job time, nodes etc) are set correctly.
#     e) make sure the time threshold for evaluating whether simulations have timed out corresponds to the cluster configurations
#     
# 4) Make sure history.txt is empty, make sure the results folder is empty, make sure that current_error_rate.txt contains one line with only the lowest error rate written in.
# 
# 5) Navigate to the directory "../d5_x/0.001/", or in a modified scenario, the folder corresponding to the lowest error rate, again using Vim or some in-terminal editor:
# 
#     a) Set the base configuration grid (fixed hyperparameters) in Generate_Base_Configs_and_Simulation_Scripts.py
#     c) Specify the variable hyper-parameter grid for this initial error rate.
#     d) Set the maximum run times for each job (each grid point will be submitted as a seperate job).
#     e) run this script with the command "python Generate_Base_Configs_and_Simulation_Scripts.py"
# 
# 6) The previous step will have generated many configuration subdirectories, as well as a "fixed_configs.p" file in the "../d5_x/" directory one level up in the directory hierachy. Check that the fixed_configs.p file has been generated. In addition check that each "config_x" subdirectory within "../d5_x/0.001/" contains:
# 
#     a) simulation_script.sh
#     b) variable_config_x.py
# 
# 7) At this stage we then have to submit all the jobs (one for each grid point) for the initial error rate. We do this by running the command "bash Start_Simulations.sh" from inside "../d5_x/0.001/".
# 
# 8) Now we have to get the script Controller.py to run periodically. Every time this script runs it will check for the current error rate and collect all available results from simulations from that error rate. If all the simulations at the specified error rate are finished, or if the time threshold for an error rate has passed, then it will write and sort the results, generate a new hyperparameter grid and simulation scripts for an increased error rate, copy the memory and weights of the optimal model from the old error rate into the appropriate directories, and submit a new batch of jobs for all the new grid points at the increased error rates. To get the controller to run periodically we do the following:
# 
#      a) Navigate into the base directory "../d5_x/" containing Controller.py
#      b) run the command "watch -n interval_in_seconds python Controller.py"
#      c) eg: for ten minute intervals: "watch -n 600 Controller.py"
# 
# 9) At this stage we are looking at the watch screen, which displays the difference in output between successive calls to Controller.py
# 
#      a) We want to detach this screen so that we can safely logout of the cluster without interrupting training
#      b) To do this type ctrl+a, then d
# 
# 10) We can now log out and the training procedure will continue safely, as directed by the Controller. We can log in and out of the cluster to see how training is proceeding whenever we want. In particular we can view the contents of:
# 
#     a) history.txt: This file contains the result of every call to Controller.py - i.e. the current error rate, how many simulations are finished or in progress, and what action was taken.
#     b) results: The results folder contains text files which contain both all the results and the best result from each error rate.
# 
# 11) When training is finished and we want to kill the controller we have to login to the cluster and run the following commands:
# 
#     a) reattach the screen with "screen -r"
#     b) We are now looking at the watch output - kill this with ctrl+c
#     c) Now we need to kill the screen with ctrl+a and then k

# In[ ]:




