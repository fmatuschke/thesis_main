#!/bin/bash

set -e

N=1250
num_p=24
sum=0
dt=1

for ((i=0; i<$N; i=i+$num_p))
do
   if [ $i -gt 400 ]; then dt=2; fi
   if [ $i -gt 800 ]; then dt=24; fi

   if [ $(($i+$num_p)) -gt $(($N-1)) ]
   then
      num_p=$((1250-$i))
      ./replace.sh cube_2pop_statistic_jureca.sh $num_p 1 $i $dt
      ((sum+=num_p))
      break
   fi
   ./replace.sh cube_2pop_statistic_jureca.sh $num_p 1 $i $dt
   ((sum+=num_p))
done

echo $sum

# ./replace.sh cube_2pop_statistic_jureca.sh 120 5 0 1:00:00
# ./replace.sh cube_2pop_statistic_jureca.sh 120 5 120 1:00:00
# ./replace.sh cube_2pop_statistic_jureca.sh 120 5 240 1:00:00
# ./replace.sh cube_2pop_statistic_jureca.sh 120 5 360 1:00:00
# ./replace.sh cube_2pop_statistic_jureca.sh 120 5 480 1:00:00
# ./replace.sh cube_2pop_statistic_jureca.sh 120 5 600 2:00:00
# ./replace.sh cube_2pop_statistic_jureca.sh 120 5 720 2:00:00
# ./replace.sh cube_2pop_statistic_jureca.sh 120 5 840 3:00:00
# ./replace.sh cube_2pop_statistic_jureca.sh 120 5 960 10:00:00
# ./replace.sh cube_2pop_statistic_jureca.sh 120 5 1080 10:00:00
# ./replace.sh cube_2pop_statistic_jureca.sh 50 3 1200 10:00:00

squeue -u matuschke1
