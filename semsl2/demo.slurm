#!/bin/bash -e
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=7:00:00
#SBATCH --job-name=jobName
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --chdir=/scratch/sca321/dlproject/semsl2
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:1

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

singularity exec --nv \
--bind /scratch \
--overlay /scratch/sca321/dlproject/data/labeled.sqsh \
--overlay /scratch/sca321/dlproject/data/unlabeled.sqsh \
--overlay /scratch/sca321/dlproject/data/conda.ext3:ro \
/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate
python threestage.py
"
