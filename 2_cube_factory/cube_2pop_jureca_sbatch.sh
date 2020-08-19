#!/bin/bash

set -e

./replace_jureca.sh cube_2pop_jureca.sh 1.0 6
# ./replace_jureca.sh cube_2pop_jureca.sh 0.5 24

squeue -u matuschke1
