#!/bin/bash

directory=$1

for d in "$directory"*/ ; do
    script_path="${d}simulation_script.sh"
    sbatch $script_path 
done