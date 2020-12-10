#!/bin/bash -l

#$ -N vgg                       # Job name
#$ -P ec523                     # Project name
#$ -o outputs/vgg               # Output file name
#$ -pe omp 2
#$ -l gpus=1
#$ -l gpu_c=6
#$ -m ea                        # Email on end or abort
#$ -j y                         # Merge output and error file

module load miniconda/4.7.5
conda activate ec523
export PYTHONPATH=/projectnb/ykh/project/Ensemble/:$PYTHONPATH

cd ..

python train.py network=vgg name=vgg
