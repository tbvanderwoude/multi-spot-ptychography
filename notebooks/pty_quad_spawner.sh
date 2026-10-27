#!/bin/bash

samples=$1
folder=$2
algo=$3
step=$((samples/4))
r=$((samples-step*4))
for d in {0..3}
do 
    if [ $d == 3 ]; then
        ./pty_script.sh $d $samples $((step*d)) $((step*(d+1)+r)) $folder $algo &
    else
        ./pty_script.sh $d $samples $((step*d)) $((step*(d+1))) $folder $algo &
    fi
done