#!/bin/bash

#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --qos=gpgpudeeplearn
#SBATCH --partition=deeplearn
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=700000
#SBATCH -A punim2715
#SBATCH --output=outputs/myjob_%j.out
#SBATCH --constraint=dlg5

# Full-node A100 booking (one entire 4-GPU node)

module purge
module load GCCcore/11.3.0
module load Apptainer/1.3.3
module load NCCL/2.21.5-CUDA-12.4.1
module load Miniconda3/23.10.0-1
module load ccache/4.6.3

source ~/projects/A100/.venv/bin/activate

jupyter notebook --no-browser --port=8890 --ip=127.0.0.1

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
