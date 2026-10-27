#!/bin/bash

samples=$1
folder=$2
algo=$3
step=$((samples/2))
r=$((samples-step*2))
for d in {2..3}
do 
    if [ $d == 3 ]; then
        ./pty_script.sh $d $samples $((step*(d-2))) $((step*(d+1-2)+r)) $folder $algo &
    else
        ./pty_script.sh $d $samples $((step*(d-2))) $((step*(d+1-2))) $folder $algo &
    fi
done