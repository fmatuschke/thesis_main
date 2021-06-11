#!/bin/bash
set -euo pipefail

NAME=cube_2pop_135_rc1
MODEL_PATH=/data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/$NAME

# single pop
/data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
   simulation_mp.py \
   -i ${MODEL_PATH}/*psi_1.00*omega_0.00_*r_0.50_*.solved.h5 \
   -o /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_single \
   --pm \
   --vervet \
   --radial \
   --single \
   -p 16
#
python3 simulation_post_0.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_single -p 16
python3 simulation_post_1.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_single -p 16
python3 simulation_post_2.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_single -p 16
#
# two flat
/data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
   simulation_mp.py \
   -i ${MODEL_PATH}/*r_0.50_*.solved.h5 \
   -o /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_flat \
   --pm \
   --vervet \
   --radial \
   --flat \
   -p 48
#
python3 simulation_post_0.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_flat -p 48
python3 simulation_post_1.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_flat -p 48
python3 simulation_post_2.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_flat -p 48
#
# Vorauswahl
/data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
   simulation_mp.py \
   -i ${MODEL_PATH}/*psi_0.[03569]0*omega_0.00*.solved.h5 \
   ${MODEL_PATH}/*psi_0.[03569]0*omega_[369]0.00*.solved.h5 \
   -o /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME} \
   --pm \
   -p 48
#
python3 simulation_post_0.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME} -p 48
python3 simulation_post_1.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME} -p 48
python3 simulation_post_2.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME} -p 48
#
# radien eliminierung
/data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
   simulation_mp.py \
   -i ${MODEL_PATH}/*r_0.50*.solved.h5 \
   -o /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_r_0.5 \
   --vervet \
   --radial \
   --pm \
   -p 48
#
python3 simulation_post_0.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_r_0.5 -p 48
python3 simulation_post_1.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_r_0.5 -p 48
python3 simulation_post_2.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/${NAME}_r_0.5 -p 48
