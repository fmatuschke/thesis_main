#!/bin/bash

set -e

./replace.sh cube_2pop_jureca.sh 1.0 6
./replace.sh cube_2pop_jureca.sh 0.5 24

squeue -u matuschke1
