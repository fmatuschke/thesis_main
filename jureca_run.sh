#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=4
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=mem512
#SBATCH --mail-user=f.matuschke@fz-juelich.de
#SBATCH --mail-type=ALL

srun -n 12 env-jureca/bin/python3 -u -m mpi4py model_2pop.py -o /p/scratch/cjinm11/matuschke1/thesis/2_cube_factory/output
srun -n 12 env-jureca/bin/python3 -u -m mpi4py simulation.py -o /p/scratch/cjinm11/matuschke1/thesis/2_cube_factory/output -i "*v0.solved*.h5"
