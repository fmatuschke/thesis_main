#!/bin/bash -x
#SBATCH --nodes=$2
#SBATCH --ntasks=$1
#SBATCH --tasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --job-name=thesis-cube_2pop_stat
#SBATCH --output=mpi.%j.out
#SBATCH --error=mpi.%j.err
#SBATCH --mail-user=f.matuschke@fz-juelich.de
#SBATCH --mail-type=ALL

# source /p/home/jusers/matuschke1/jureca/private/thesis/jureca_modules.sh
# srun -n 240 /p/home/jusers/matuschke1/jureca/private/thesis/env/bin/python -m mpi4py cube_2pop_statistic_jureca.py -o /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_stat_ -n 10000 --start 0
# srun -n 240 /p/home/jusers/matuschke1/jureca/private/thesis/env/bin/python -m mpi4py cube_2pop_statistic_jureca.py -o /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_stat_ -n 10000 --start 240
# srun -n 240 /p/home/jusers/matuschke1/jureca/private/thesis/env/bin/python -m mpi4py cube_2pop_statistic_jureca.py -o /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_stat_ -n 10000 --start 480
# srun -n 240 /p/home/jusers/matuschke1/jureca/private/thesis/env/bin/python -m mpi4py cube_2pop_statistic_jureca.py -o /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_stat_ -n 10000 --start 720
# srun -n 240 /p/home/jusers/matuschke1/jureca/private/thesis/env/bin/python -m mpi4py cube_2pop_statistic_jureca.py -o /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_stat_ -n 10000 --start 960
# srun -n 240 /p/home/jusers/matuschke1/jureca/private/thesis/env/bin/python -m mpi4py cube_2pop_statistic_jureca.py -o /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_stat_ -n 10000 --start 1200

srun -n $1 /p/home/jusers/matuschke1/jureca/private/thesis/env/bin/python3 -m mpi4py cube_2pop_statistic_jureca.py -o /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_stat_ -n 10000 --start $3
