#!/bin/bash -l

#$ -N ensemble			# Job name
#$ -P ec523			# Project name
#$ -o ../outputs/ensemble	# Output file name
#$ -pe omp 2		
#$ -l gpus=1 		
#$ -l gpu_c=6 	
#$ -m ea			# Email on end or abort
#$ -j y				# Merge output and error file
 
module load miniconda/4.7.5
conda activate ec523
export PYTHONPATH=/projectnb/ykh/project/Ensemble/:$PYTHONPATH

python ../train.py type=ensemble name=ensemble n_epochs=1 subnet_type=tuned sub1_path='../checkpoints/sub1_tuned/epoch_300' sub2_path='../checkpoints/sub2_tuned/epoch_300' sub3_path='../checkpoints/sub3_tuned/epoch_300'
