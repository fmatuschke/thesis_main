#!/bin/bash

set -e

./replace.sh cube_2pop_statistic_jureca.sh 120 5 0 0:10:00
./replace.sh cube_2pop_statistic_jureca.sh 120 5 120 0:10:00
./replace.sh cube_2pop_statistic_jureca.sh 120 5 240 0:10:00
./replace.sh cube_2pop_statistic_jureca.sh 120 5 360 0:30:00
./replace.sh cube_2pop_statistic_jureca.sh 120 5 480 0:30:00
./replace.sh cube_2pop_statistic_jureca.sh 120 5 600 1:00:00
./replace.sh cube_2pop_statistic_jureca.sh 120 5 720 1:00:00
./replace.sh cube_2pop_statistic_jureca.sh 120 5 840 2:00:00
./replace.sh cube_2pop_statistic_jureca.sh 120 5 960 5:00:00
./replace.sh cube_2pop_statistic_jureca.sh 120 5 1080 10:00:00
./replace.sh cube_2pop_statistic_jureca.sh 50 3 1200 10:00:00

squeue -u matuschke1
