#!/bin/bash -x
#SBATCH --nodes=8
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=4
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=mem512
#SBATCH --mail-user=f.matuschke@fz-juelich.de
#SBATCH --mail-type=ALL

source jureca_modules.sh
srun -n 96 env-jureca/bin/python3 -u -m mpi4py simulation.py -t 4 --input /p/scratch/cjinm11/matuschke1/thesis/2/output/models/*.solved.h5 --output /p/scratch/cjinm11/matuschke1/thesis/5/output 
