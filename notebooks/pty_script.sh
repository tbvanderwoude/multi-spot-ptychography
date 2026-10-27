#!/bin/bash
device=$1
samples=$2
start=$3
end=$4
folder=$5
algo=$6

CUDA_VISIBLE_DEVICES=$1 python pty_simulation.py --device=0 --N_samples=$2 --start_index=$3 --end_index=$4 --folder=$5 --algo=$6