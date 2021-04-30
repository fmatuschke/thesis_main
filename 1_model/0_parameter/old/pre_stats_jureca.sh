#!/bin/bash -x
#SBATCH --nodes=$2
#SBATCH --ntasks=$1
#SBATCH --tasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --time=$4:00:00
#SBATCH --partition=batch
#SBATCH --job-name=thesis-cube_2pop_stat
#SBATCH --output=mpi.%j.out
#SBATCH --error=mpi.%j.err
#SBATCH --mail-user=f.matuschke@fz-juelich.de
#SBATCH --mail-type=ALL

THESIS="$(git rev-parse --show-toplevel)"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
NAME="$(basename $0 | rev | cut -d'_' -f 1 | rev)"

source $THESIS/jureca_modules.sh

Num=1
OMP_NUM_THREADS=$Num
export OMP_NUM_THREADS

srun -n $1 $THESIS/env-jureca/bin/python3 \
   -m mpi4py parameter_statistic.py \
   -o /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_stat_ \
   -n 100000 \
   --start $3 \
   --time $4 \
   -p $Num
