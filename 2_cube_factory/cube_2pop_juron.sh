#!/bin/bash -x
#BSUB -n 100
#BSUB -R "span[ptile=25]"
#BSUB -W 24:00
#BSUB -q normal
#BSUB -J thesis-cube_2pop
#BSUB -o "stdout.%J.out"
#BSUB -e "stderr.%J.out"

source /p/home/jusers/matuschke1/juron/private/thesis/juron_modules.sh

srun -n 100 /p/project/cjinm11/Private/matuschke1/thesis/env-juron/bin/python3 \
   -u -m mpi4py cube_2pop.py \
   -o /p/scratch/cjinm11/matuschke1/thesis/2/output/cube_2pop_0/ \
   -r 1.0 -v 210 -n 10000 -p 4

# srun -n 100 /p/project/cjinm11/Private/matuschke1/thesis/env-juron/bin/python3 \
#    -u -m mpi4py cube_2pop.py \
#    -o /p/scratch/cjinm11/matuschke1/thesis/2/output/cube_2pop_0/ \
#    -r 0.5 -v 105 -n 10000 -p 4
