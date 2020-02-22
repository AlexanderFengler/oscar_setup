#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J gpu_test

# priority
##SBATCH --account=bibs-frankmj-condo
#SBATCH --account=carney-frankmj-condo

# email error reports
#SBATCH --mail-user=alexander_fengler@brown.edu 
#SBATCH --mail-type=ALL

# output file
#SBATCH --output /users/afengler/batch_job_out/gpu_test.out

# Request runtime, memory, cores
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH -c 14
#SBATCH -N 1
#SBATCH --constraint='quadrortx'
##SBATCH --constraint='cascade'
#SBATCH -p gpu --gres=gpu:1
##SBATCH --array=1-300

conda deactivate
conda activate tf-gpu-py37
# module load python/3.7.4 cuda/10.0.130 cudnn/7.4 tensorflow/2.0.0_gpu_py37
python -u /users/afengler/git_repos/oscar_setup/keras_example.py