#!/bin/bash -x
#BSUB -n 10
#BSUB -x
#BSUB -R "span[ptile=2]"
#BSUB -W 24:00
#BSUB -q normal
#BSUB -J thesis-sim_vs
#BSUB -o "stdout.%J.out"
#BSUB -e "stderr.%J.out"
#BSUB -u f.matuschke@fz-juelich.de
#BSUB -B
#BSUB -N

source /p/home/jusers/matuschke1/juron/private/thesis/juron_modules.sh

Num=10
OMP_NUM_THREADS=$Num
export OMP_NUM_THREADS

mpirun /p/home/jusers/matuschke1/juron/private/thesis/env-juron/bin/python3 voxel_size.py -i /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/cube_2pop_psi_1.00_omega_0.00_r_*_v0_120_.solved.h5 /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/cube_2pop_psi_0.50_omega_90.00_r_*_v0_120_.solved.h5 -o /p/scratch/cjinm11/matuschke1/thesis/2/0/vs_120 -p 1 -t $Num -m 11 -n 10

# mpirun /p/home/jusers/matuschke1/juron/private/thesis/env-juron/bin/python3 \
# 	-m mpi4py parameter_statistic.py \
# 	-o /p/scratch/cjinm11/matuschke1/thesis/2/$5 \
# 	-n 100000 \
# 	--start $3 \
# 	--time $4 \
# 	-p $Num
