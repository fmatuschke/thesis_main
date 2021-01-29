#!/bin/bash -x

#  # Vorauswahl
mpirun -n 48 /data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
   -m mpi4py simulation_ime.py \
   -i /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.00*omega_0.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.30*omega_0.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.30*omega_30.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.30*omega_60.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.30*omega_90.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.50*omega_0.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.50*omega_30.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.50*omega_60.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.50*omega_90.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.60*omega_0.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.60*omega_30.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.60*omega_60.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.60*omega_90.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.90*omega_0.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.90*omega_30.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.90*omega_60.00*.solved.h5 \
   /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*psi_0.90*omega_90.00*.solved.h5 \
   -o /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/sim_120 \
   -v 0.125 \
   --start 0 \
   --n_inc 4 \
   --d_rot 15
#
#
# # # radien eliminierung
# mpirun -n 48 /data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
#    -m mpi4py simulation_ime.py \
#    -i /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*omega_0.00*r_0.50*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*omega_30.00*r_0.50*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*omega_60.00*r_0.50*.solved.h5 \
#    /data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/*omega_90.00*r_0.50*.solved.h5 \
#    -o /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/sim_120_ime_radius_0.5_ \
#    -v 0.125 \
#    --start 0
