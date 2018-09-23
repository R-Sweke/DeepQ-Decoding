# I have to write a script which goes and makes all the start simulation scripts executable!!!

#!/bin/bash

for d in */ ; do
    file_path="./${d}Start_Continuing_Simulations.sh"
    echo $file_path
    chmod +x $file_path
done