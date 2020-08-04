#!/bin/bash

set -e

./replace.sh cube_2pop_statistic_jureca.sh 120 5 0 1
./replace.sh cube_2pop_statistic_jureca.sh 120 5 120 1
./replace.sh cube_2pop_statistic_jureca.sh 120 5 240 1
./replace.sh cube_2pop_statistic_jureca.sh 120 5 360 1
./replace.sh cube_2pop_statistic_jureca.sh 120 5 480 1
./replace.sh cube_2pop_statistic_jureca.sh 120 5 600 1
./replace.sh cube_2pop_statistic_jureca.sh 120 5 720 1
./replace.sh cube_2pop_statistic_jureca.sh 120 5 840 5
./replace.sh cube_2pop_statistic_jureca.sh 120 5 960 5
./replace.sh cube_2pop_statistic_jureca.sh 120 5 1080 10
./replace.sh cube_2pop_statistic_jureca.sh 50 3 1200 10

squeue -u matuschke1
