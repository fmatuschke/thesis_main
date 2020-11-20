#!/bin/bash -x
#BSUB -n $1
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

Num=1
OMP_NUM_THREADS=$Num
export OMP_NUM_THREADS

mpirun /p/project/cjinm11/Private/matuschke1/thesis/env-juron/bin/python3 \
   -u -m mpi4py cube_2pop.py \
   -o /p/scratch/cjinm11/matuschke1/thesis/2/$4 \
   -r $3 -v 120 -n 100000 -p $Num --start $2 --time 24
