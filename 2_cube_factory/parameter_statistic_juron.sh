#!/bin/bash -x
#BSUB -n $1
#BSUB -W $4:00
#BSUB -q normal
#BSUB -J thesis-cube_2pop_stat
#BSUB -o "stdout.%J.out"
#BSUB -e "stderr.%J.out"

source /p/home/jusers/matuschke1/juron/private/thesis/juron_modules.sh

mpirun /p/home/jusers/matuschke1/juron/private/thesis/env-juron/bin/python3 \
				-m mpi4py parameter_statistic.py \
				-o /p/scratch/cjinm11/matuschke1/thesis/2/cube_2pop_stat_juron_ \
			   -n 10000 \
				--start $3 \
		   	--time $4
