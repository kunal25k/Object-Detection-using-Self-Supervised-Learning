#!/bin/bash 
#SBATCH --nodes=1            # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=0:10:00
#SBATCH --mem=6GB
#SBATCH --job-name=demo
#SBATCH --output=demo.out
#SBATCH --gres=gpu:1

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

module load python/intel/3.8.6

singularity exec --nv \
--bind /scratch \
--overlay labeled.sqsh \
/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "

python demo.py
"

