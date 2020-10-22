#!/bin/bash

set -e

THESIS="$(git rev-parse --show-toplevel)"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# NAME="$(basename $0 | rev | cut -d'_' -f2- | rev)"

echo $NAME

mpirun -n 2 $THESIS/env-ime263/bin/python3 \
   -m mpi4py $DIR/pre_stats.py \
   -o $DIR/output/pre_stats_test \
   -n 100 \
   --start 0 \
   --time 1 \
   -p 2
