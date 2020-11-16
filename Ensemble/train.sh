#!/bin/bash -l

#$ -N sub3_tuned	# Job name
#$ -P ec523		# Project name
#$ -o output_sub3_tuned	# Output file name
#$ -pe omp 2		
#$ -l gpus=1 		
#$ -l gpu_c=6 	
#$ -m ea		# Email on end or abort
#$ -j y			# Merge output and error file
 
module load miniconda/4.7.5
conda activate ec523
export PYTHONPATH=/projectnb/ykh/project/Ensemble/:$PYTHONPATH

python train.py name=sub3_tuned n_epochs=300 
