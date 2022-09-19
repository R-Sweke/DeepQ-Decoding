# I have to write a script which goes and makes all the start simulation scripts executable!!!

#!/bin/bash

for d in */ ; do
    file_path="./${d}Start_Continuing_Simulations.sh"
    if [ -f $file_path ]; then
        echo $file_path
        chmod +x $file_path
    fi
done