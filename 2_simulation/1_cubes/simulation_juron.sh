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

# python3 simulation_juron.py -i ../../1_model/1_cubes/output/cube_2pop_120/*omega_0.00*.solved.h5 ../../1_model/1_cubes/output/cube_2pop_120/*omega_30.00*.solved.h5 ../../1_model/1_cubes/output/cube_2pop_120/*omega_60.00*.solved.h5 ../../1_model/1_cubes/output/cube_2pop_120/*omega_90.00*solved.h5 -o output/test -v 1 --start 0

mpirun /p/home/jusers/matuschke1/juron/private/thesis/env-juron/bin/python3 \
   -m mpi4py simulation_juron.py \
   -i /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*omega_0.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*omega_30.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*omega_60.00*.solved.h5 \
   /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_120/*omega_90.00*.solved.h5 \
   -o /p/scratch/cjinm11/matuschke1/thesis/2/sim_120_new/$1 \
   -v $1 \
   --start $3
