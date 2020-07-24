#!/bin/bash -x
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=24
#SBATCH --time=5:00:00
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --job-name=thesis-5
#SBATCH --output=mpi.%j.out
#SBATCH --error=mpi.%j.err
#SBATCH --mail-user=f.matuschke@fz-juelich.de
#SBATCH --mail-type=ALL

srun -n 10 ../env-jureca/bin/python voxel_size_simulation_jureca.py -i /p/fastdata/pli/Projects/Felix/thesis/data/models/1_rnd_seed/cube_2pop_psi_1.00_omega_0.00_r_1.00_v0_210_.solved.h5 /p/fastdata/pli/Projects/Felix/thesis/data/models/1_rnd_seed/cube_2pop_psi_0.50_omega_90.00_r_1.00_v0_210_.solved.h5 -o /p/scratch/cjinm11/matuschke1/thesis/5/repeat_test_0 -t 24 -m 51 -n 10
