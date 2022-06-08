#!/bin/bash
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --time=24:0:0 

idx=$1
python blink/main_dense_new.py -i --fast --doc_start_index $idx
