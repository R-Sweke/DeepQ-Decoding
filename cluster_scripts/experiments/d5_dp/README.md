Before beginning make sure that this directory (now on referred to as the "base" directory "./") contains:

   - Controller.py
   - make_executable.sh
   - static_decoder (an appropriate referee decoder with the corresponding lattice size and error model)
   - An empty folder called "results"
   - An empty text document called "history.txt"
   - A subdirectory for each error rate one would like to iterate through
   - A text file "current_error_rate.txt" containing one line with the lowest error rate - i.e. 0.001
    
Furthermore, the subdirectory corresponding to the lowest error rate (0.001 here) should contain:

   - Generate_Base_Configs_and_Simulation_Scripts.py
   - Single_Point_Training_Script.py
   - Start_Simulations.sh
   - An empty folder "output_files"
    
Additionally every other error rate subdirectory should contain:

   - Single_Point_Continue_Training_Script.py
   - Start_Continuing_Simulations.sh
   - An empty folder "output_files"


In order to run a customized/modified version of this procedure this exact directory and file structure should be replicated, as we will see below all that is necessary is to modify:

   - Controller.py
   - Generate_Base_Configs_and_Simulation_Scripts.py
   - Providing the appropriate static/referee decoder
   - Renaming the error rate subdirectories appropriately

To begin, copy the entire base directory onto the HPC cluster and navigate into the base directory. Then:

1) From inside the base directory start a "screen" - this provides a persistent terminal which we can later detach and re-attach at will to keep track of the training procedure, without having to remain logged in to the cluster.

     a) type "screen"
     b) then press enter
     c) We are now in a screen
     
2) Run the command "bash make_executable.sh". This will allow the controller - a python script which will be run periodically to control the training process - to submit jobs via slurm.

3) Using vim or some other in-terminal editor, modify the following in Controller.py:
    
    a) Set all the error rates that you would like to iterate through - make sure there is an appropriate subdirectory for each error rate given here.
    b) For each error rate, provide the expected lifetime of a single faulty qubit (i.e. the threshold for decoding sucess) as well as the average qubit lifetime you would like to use as a threshold for stopping training. We recommend setting this training threshold extremely high, so that training ends due to convergence.
    c) set the hyper-parameter grid that you would like to use at each error rate iteration.
    d) Also make sure that all the cluster parameters (job time, nodes etc) are set correctly.
    e) make sure the time threshold for evaluating whether simulations have timed out corresponds to the cluster configurations
    
4) Make sure history.txt is empty, make sure the results folder is empty, make sure that current_error_rate.txt contains one line with only the lowest error rate written in.

5) Navigate to the directory "./0.001/", or in a modified scenario, the folder corresponding to the lowest error rate, again using Vim or some in-terminal editor:

    a) Set the base configuration grid (fixed hyperparameters) in Generate_Base_Configs_and_Simulation_Scripts.py
    c) Specify the variable hyper-parameter grid for this initial error rate.
    d) Set the maximum run times for each job (each grid point will be submitted as a seperate job).
    e) run this script with the command "python Generate_Base_Configs_and_Simulation_Scripts.py"

6) The previous step will have generated many configuration subdirectories, as well as a "fixed_configs.p" file in the base directory one level up in the directory hierachy. Check that the fixed_configs.p file has been generated. In addition check that each "config_x" subdirectory within "../d5_x/0.001/" contains:

    a) simulation_script.sh
    b) variable_config_x.py

7) At this stage we then have to submit all the jobs (one for each grid point) for the initial error rate. We do this by running the command "bash Start_Simulations.sh" from inside "./0.001/".

8) Now we have to get the script Controller.py to run periodically. Every time this script runs it will check for the current error rate and collect all available results from simulations from that error rate. If all the simulations at the specified error rate are finished, or if the time threshold for an error rate has passed, then it will write and sort the results, generate a new hyperparameter grid and simulation scripts for an increased error rate, copy the memory and weights of the optimal model from the old error rate into the appropriate directories, and submit a new batch of jobs for all the new grid points at the increased error rates. To get the controller to run periodically we do the following:

     a) Navigate into the base directory containing Controller.py
     b) run the command "watch -n interval_in_seconds python Controller.py"
     c) eg: for ten minute intervals: "watch -n 600 Controller.py"

9) At this stage we are looking at the watch screen, which displays the difference in output between successive calls to Controller.py

     a) We want to detach this screen so that we can safely logout of the cluster without interrupting training
     b) To do this type ctrl+a, then d

10) We can now log out and the training procedure will continue safely, as directed by the Controller. We can log in and out of the cluster to see how training is proceeding whenever we want. In particular we can view the contents of:

    a) history.txt: This file contains the result of every call to Controller.py - i.e. the current error rate, how many simulations are finished or in progress, and what action was taken.
    b) results: The results folder contains text files which contain both all the results and the best result from each error rate.

11) When training is finished and we want to kill the controller we have to login to the cluster and run the following commands:

    a) reattach the screen with "screen -r"
    b) We are now looking at the watch output - kill this with ctrl+c
    c) Now we need to kill the screen with ctrl+a and then k