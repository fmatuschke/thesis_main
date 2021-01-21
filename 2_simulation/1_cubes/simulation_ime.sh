#!/bin/bash -x

#  # Vorauswahl
# mpirun -n 48 /data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
#    -m mpi4py simulation_ime.py \
# -i /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.00*omega_0.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.30*omega_0.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.30*omega_30.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.30*omega_60.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.30*omega_90.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.50*omega_0.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.50*omega_30.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.50*omega_60.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.50*omega_90.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.60*omega_0.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.60*omega_30.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.60*omega_60.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.60*omega_90.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.90*omega_0.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.90*omega_30.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.90*omega_60.00*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.90*omega_90.00*.solved.h5
#    -o /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/sim_120_ime_ \
#    -v 0.125 \
#    --start 0
#
#
# # radien eliminierung
mpirun -n 48 /data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
   -m mpi4py simulation_ime.py \
   -i /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*r_1.00*.solved.h5 \
   -o /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/sim_120_ime_radius_1 \
   -v 0.125 \
   --start 0
