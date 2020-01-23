#!/bin/bash -x
#SBATCH --nodes=8
#SBATCH --ntasks=92
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=4
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=4:00:00
#SBATCH --partition=batch
#SBATCH --mail-user=f.matuschke@fz-juelich.de
#SBATCH --mail-type=ALL

srun -n 92 env-jureca/bin/python3 -u model_2pop.py /p/scratch/cjinm11/matuschke1/thesis/2_cube_factory/output
srun -n 92 env-jureca/bin/python3 -u simulation.py /p/scratch/cjinm11/matuschke1/thesis/2_cube_factory/output
