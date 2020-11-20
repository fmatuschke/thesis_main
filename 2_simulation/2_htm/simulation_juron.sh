#!/bin/bash -x
#BSUB -n 20
#BSUB -W 24:00
#BSUB -q normal
#BSUB -J thesis-htm_sim
#BSUB -o "stdout.%J.out"
#BSUB -e "stderr.%J.out"

source /p/home/jusers/matuschke1/juron/private/thesis/juron_modules.sh

Num=1
OMP_NUM_THREADS=$Num
export OMP_NUM_THREADS

mpirun /p/home/jusers/matuschke1/juron/private/thesis/env-juron/bin/python3 \
            -m mpi4py simulation.py \
            -i /p/scratch/cjinm11/matuschke1/thesis/1/htm_60/*.solved.h5 \
            -o /p/scratch/cjinm11/matuschke1/thesis/2/htm_60/$1 \
            -v $1 \
            --start $2
