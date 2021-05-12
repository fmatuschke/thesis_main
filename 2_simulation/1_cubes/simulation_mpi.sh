#!/bin/bash
set -euo pipefail

NAME=cube_2pop_135_rc1
MODEL_PATH=/data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/$NAME

# Vorauswahl
mpirun -n 48 /data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
   -m mpi4py.futures simulation_mpi.py \
   -i ${MODEL_PATH}/*psi_0.[03569]0*omega_0.00*.solved.h5 \
   ${MODEL_PATH}/*psi_0.[03569]0*omega_[369]0.00*.solved.h5 \
   -o /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME} \
   --pm
#
python3 simulation_post_0.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME} -p 48
python3 simulation_post_1.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME} -p 48
python3 simulation_post_2.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME} -p 48
#
# radien eliminierung
mpirun -n 48 /data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
   -m mpi4py.futures simulation_mpi.py \
   -i ${MODEL_PATH}/*r_0.50*.solved.h5 \
   -o /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_r_0.5 \
   --Vervet \
   --radial \
   --pm
#
python3 simulation_post_0.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_r_0.5 -p 48
python3 simulation_post_1.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_r_0.5 -p 48
python3 simulation_post_2.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_r_0.5 -p 48
