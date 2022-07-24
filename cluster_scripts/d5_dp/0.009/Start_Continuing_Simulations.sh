#!/bin/bash

directory=$1

for d in "$directory"*/ ; do
    script_path="${d}simulation_script.sh"
    if [ -f $script_path ]; then
        sbatch $script_path
    fi 
done