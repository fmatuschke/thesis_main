#!/bin/bash

set -e

./replace.sh cube_2pop_statistic_jureca.sh 120 5 0
./replace.sh cube_2pop_statistic_jureca.sh 120 5 120
./replace.sh cube_2pop_statistic_jureca.sh 120 5 240
./replace.sh cube_2pop_statistic_jureca.sh 120 5 360
./replace.sh cube_2pop_statistic_jureca.sh 120 5 480
./replace.sh cube_2pop_statistic_jureca.sh 120 5 600
./replace.sh cube_2pop_statistic_jureca.sh 120 5 720
./replace.sh cube_2pop_statistic_jureca.sh 120 5 840
./replace.sh cube_2pop_statistic_jureca.sh 120 5 960
./replace.sh cube_2pop_statistic_jureca.sh 120 5 1080
./replace.sh cube_2pop_statistic_jureca.sh 50 3 1200

squeue -u matuschke1
