#!/bin/bash -x
set -euo pipefail

NAME=sim_cube_2pop_120
MODEL_PATH=/data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120

# Vorauswahl
mpirun -n 48 /data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
   -m mpi4py simulation_ime.py \
   -i ${MODEL_PATH}/*psi_0.[03569]0*omega_0.00*.solved.h5 \
   ${MODEL_PATH}/*psi_0.[03569]0*omega_[369]0.00*.solved.h5 \
   -o /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME} \
   -v 0.125 \
   --start 0 \
   --n_inc 4 \
   --d_rot 15
#
#
# radien eliminierung
mpirun -n 48 /data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
   -m mpi4py simulation_ime.py \
   -i ${MODEL_PATH}/*r_0.50*.solved.h5 \
   -o /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_r_0.5 \
   -v 0.125 \
   --start 0 \
   --n_inc 4 \
   --d_rot 15 \
   --Vervet \
   --radial
