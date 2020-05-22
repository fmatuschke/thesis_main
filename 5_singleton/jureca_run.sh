#!/bin/bash -x
#SBATCH --nodes=8
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=4
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=04:00:00
#SBATCH --partition=gpus
#SBATCH --gres=gpu:2
#SBATCH --mail-user=f.matuschke@fz-juelich.de
#SBATCH --mail-type=ALL

source jureca_modules.sh
srun -n 96 env-jureca/bin/python3 -u -m mpi4py simulation.py -t 4 --input /p/fastdata/pli/Projects/Felix/thesis/models/*.solved.h5 \
                                                                  --output /p/scratch/cjinm11/matuschke1/thesis/5/output/12.5 \
                                                                  --tilting --voxel-size 0.025 0.05 0.1 0.25 0.75 1.25 2.5 --length 12.5
                                                                  
srun -n 96 env-jureca/bin/python3 -u -m mpi4py simulation.py -t 4 --input /p/fastdata/pli/Projects/Felix/thesis/models/*.solved.h5 \
                                                                  --output /p/scratch/cjinm11/matuschke1/thesis/5/output/60.0 \
                                                                  --tilting --voxel-size 0.1 0.25 0.75 1.25 2.5 --length 60.0                                                               
                                                                  