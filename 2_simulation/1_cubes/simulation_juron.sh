#!/bin/bash -x
#BSUB -n 20
#BSUB -W 10:00
#BSUB -q normal
#BSUB -J thesis-cube_2pop_sim
#BSUB -o "stdout.%J.out"
#BSUB -e "stderr.%J.out"

source /p/home/jusers/matuschke1/juron/private/thesis/juron_modules.sh

Num=1
OMP_NUM_THREADS=$Num
export OMP_NUM_THREADS

mpirun /p/home/jusers/matuschke1/juron/private/thesis/env-juron/bin/python3 \
   -m mpi4py simulation_juron.py \
   -i /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.00*omega_0.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.30*omega_0.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.30*omega_30.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.30*omega_60.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.30*omega_90.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.50*omega_0.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.50*omega_30.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.50*omega_60.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.50*omega_90.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.60*omega_0.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.60*omega_30.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.60*omega_60.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.60*omega_90.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.90*omega_0.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.90*omega_30.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.90*omega_60.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*psi_0.90*omega_90.00*.solved.h5 \
   -o /p/scratch/cjinm11/matuschke1/thesis/2/sim_120_new/$1 \
   -v $1 \
   --start $3
