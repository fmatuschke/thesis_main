#!/bin/bash -x
#BSUB -n $1
#BSUB -W $4:00
#BSUB -q normal
#BSUB -J thesis-cube_2pop_stat
#BSUB -o "stdout.%J.out"
#BSUB -e "stderr.%J.out"

source /p/home/jusers/matuschke1/juron/private/thesis/juron_modules.sh

Num=1
OMP_NUM_THREADS=$Num
export OMP_NUM_THREADS

mpirun /p/home/jusers/matuschke1/juron/private/thesis/env-juron/bin/python3 \
				-m mpi4py parameter_statistic.py \
				-o /p/scratch/cjinm11/matuschke1/thesis/2/$5 \
			   -n 100000 \
				--start $3 \
		   	--time $4 \
				-p $Num
