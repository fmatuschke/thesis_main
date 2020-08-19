#!/bin/bash -x
#SBATCH --nodes=4
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=$2:00:00
#SBATCH --partition=batch
#SBATCH --mail-user=f.matuschke@fz-juelich.de
#SBATCH --mail-type=ALL

srun -n 96 /p/project/cjinm11/Private/matuschke1/thesis/env-jureca/bin/python3 \
   -u -m mpi4py cube_2pop_sc.py \
   -o /p/scratch/cjinm11/matuschke1/thesis/2/output/cube_2pop_2/ \
   -r $1 -v 90 -n 10000 -p 1
