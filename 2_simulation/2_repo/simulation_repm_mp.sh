#!/bin/bash
set -euo pipefail

NAME=cube_2pop_135_repm_rc1
MODEL_PATH=/data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_135_rc1

# single pop
/data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
   simulation_repm_mp.py \
   -i ${MODEL_PATH}/*psi_1.00*omega_0.00_*r_*.solved.h5 \
   -o output/${NAME}_single \
   --pm \
   --vervet \
   --radial \
   --single \
   -m 50 \
   -p 10
#
# two flat
/data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
   simulation_repm_mp.py \
   -i ${MODEL_PATH}/*psi_0.50*omega_90.00_*r_*.solved.h5 \
   -o output/${NAME}_flat \
   --pm \
   --vervet \
   --radial \
   --flat \
   -m 50 \
   -p 10
