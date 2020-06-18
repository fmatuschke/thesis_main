#!/bin/bash -x
#SBATCH --nodes=8
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=4
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --mail-user=f.matuschke@fz-juelich.de
#SBATCH --mail-type=ALL

srun -n 96 /p/project/cjinm11/Private/matuschke1/thesis/env/bin/python3 -u -m mpi4py cube_2pop.py -o /p/scratch/cjinm11/matuschke1/thesis/2/output/cube_2pop_0/ -r 1.0 -n 10000 -p 4
