#!/usr/bin/env bash
# Based on A100-fullnode.sh. Submit through A100-ConnectX6.sh.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:4
#SBATCH --qos=gpgpudeeplearn
#SBATCH --partition=deeplearn
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=700000
#SBATCH --account=punim2715
#SBATCH --constraint=dlg5
#SBATCH --exclusive
#SBATCH --output=outputs/connectx6-booking/job_%j.out

BOOKING_DIR=/data/gpfs/projects/punim2715/bookings
cd "$BOOKING_DIR" || exit 1
JUPYTER_PID=
cleanup() {
    [[ -z "$JUPYTER_PID" ]] || kill "$JUPYTER_PID" 2>/dev/null || true
    exit 0
}
trap cleanup TERM INT

# Preserve the original notebook setup; setup/notebook failure must not end
# the booking. The parent below stays alive until Slurm ends the allocation.
(
    set -e
    source /etc/profile
    module purge
    module load GCCcore/11.3.0 Apptainer/1.3.3 NCCL/2.21.5-CUDA-12.4.1 Miniconda3/23.10.0-1 ccache/4.6.3
    source ~/projects/A100/.venv/bin/activate
    exec jupyter notebook --no-browser --port=8890 --ip=127.0.0.1
) &
JUPYTER_PID=$!
while true; do sleep 300 & wait "$!"; done
