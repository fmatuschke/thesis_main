#!/bin/bash -x
#SBATCH --nodes=2
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=12
#SBATCH --cpu-per-task=2
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=10:00:00
#SBATCH --partition=batch
#SBATCH --mail-user=f.matuschke@fz-juelich.de
#SBATCH --mail-type=ALL

srun -n 24 env-jureca/bin/python3 -u model_2pop.py /p/scratch/cjinm11/matuschke1/thesis/2_cube_factory
