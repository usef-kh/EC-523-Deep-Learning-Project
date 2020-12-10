#!/bin/bash -l

#$ -N ensemble                  # Job name
#$ -P ec523                     # Project name
#$ -o outputs/ensemble          # Output file name
#$ -pe omp 2
#$ -l gpus=1
#$ -l gpu_c=6
#$ -m ea                        # Email on end or abort
#$ -j y                         # Merge output and error file

module load miniconda/4.7.5
conda activate ec523
export PYTHONPATH=/projectnb/ykh/project/Ensemble/:$PYTHONPATH

cd ..

python train.py network=ensemble name=ensemble n_epochs=3 subnet_type=tuned sub1_path='checkpoints/fer2013/sub1/epoch_300' sub2_path='checkpoints/fer2013/sub2/epoch_300' sub3_path='checkpoints/fer2013/sub3/epoch_300' vgg_path='checkpoints/fer2013/vgg/epoch_300'
