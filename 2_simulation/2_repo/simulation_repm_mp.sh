#!/bin/bash
set -euo pipefail

NAME=cube_2pop_135_repm_rc1
MODEL_PATH=/data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_135_rc1

# single pop
/data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
   simulation_repm_mp.py \
   -i ${MODEL_PATH}/*psi_1.00*omega_0.00_*r_*.solved.h5 \
   -o /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_single \
   --pm \
   --vervet \
   --radial \
   --single \
   -m 5 \
   -p 16
#
# # two flat
# /data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
#    simulation_repm_mp.py \
#    -i ${MODEL_PATH}/*r_0.50_*.solved.h5 \
#    -o /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_flat \
#    --pm \
#    --vervet \
#    --radial \
#    --flat \
#    -m 5 \
#    -p 48
# #
# # python3 simulation_post_0.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_flat -p 48
# # python3 simulation_post_1.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_flat -p 48
# # python3 simulation_post_2.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_flat -p 48
