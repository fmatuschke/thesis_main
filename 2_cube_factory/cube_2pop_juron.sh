#!/bin/bash -x
#BSUB -n 80
#BSUB -x
#BSUB -R "span[ptile=20]"
#BSUB -W 24:00
#BSUB -q normal
#BSUB -J thesis-cube_2pop
#BSUB -o "stdout.%J.out"
#BSUB -e "stderr.%J.out"
#BSUB -u f.matuschke@fz-juelich.de
#BSUB -B
#BSUB -N

source /p/home/jusers/matuschke1/juron/private/thesis/juron_modules.sh

Num=8
OMP_NUM_THREADS=$Num
export OMP_NUM_THREADS

# --map-by ppr:20:node

mpirun /p/project/cjinm11/Private/matuschke1/thesis/env-juron/bin/python3 \
   -u -m mpi4py cube_2pop.py \
   -o /p/scratch/cjinm11/matuschke1/thesis/2/output/cube_2pop_juron/ \
   -r 1.0 -v 210 -n 10000 -p $Num

# mpirun -n 80 /p/project/cjinm11/Private/matuschke1/thesis/env-juron/bin/python3 \
#    -u -m mpi4py cube_2pop.py \
#    -o /p/scratch/cjinm11/matuschke1/thesis/2/output/cube_2pop_juron_test/ \
#    -r 0.5 -v 105 -n 10000 -p $Num
