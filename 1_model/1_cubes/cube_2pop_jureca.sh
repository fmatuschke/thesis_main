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

source module /p/project/cjinm11/Private/matuschke1/thesis/jureca_modules.sh

srun -n 96 /p/project/cjinm11/Private/matuschke1/thesis/env-jureca/bin/python3 \
   -u -m mpi4py /p/project/cjinm11/Private/matuschke1/thesis/2_cube_factory/cube_2pop.py \
   -o /p/scratch/cjinm11/matuschke1/thesis/2/output/$3 \
   -r $1 -v 90 -n 100000 -p 1 --start 0
