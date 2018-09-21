#!/bin/bash

for d in */ ; do
    script_path="./${d}simulation_script.sh"
    sbatch $script_path
done