#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=mem512
#SBATCH --mail-user=f.matuschke@fz-juelich.de
#SBATCH --mail-type=ALL

env-jureca/bin/python3 -u run.py -n 48 -t 4 --all --output /p/scratch/cjinm11/matuschke1/thesis/5/output --input /p/fastdata/pli/Projects/Felix/thesis/models/*.solved.h5
