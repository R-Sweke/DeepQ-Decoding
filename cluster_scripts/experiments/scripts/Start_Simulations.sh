#!/bin/bash

for d in */ ; do
    script_path="./${d}simulation_script.sh"
    if [ -f $script_path ]; then
        sbatch $script_path
    fi
done