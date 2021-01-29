#!/bin/bash -x

mpirun -n 48 \
   /data/PLI-Group/felix/data/thesis/env-$(hostname)/bin/python3 \
   -m mpi4py generate_tissue.py \
   -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/sim_120_ime
