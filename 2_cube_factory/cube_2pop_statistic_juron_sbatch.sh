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
      ./juron_replace.sh cube_2pop_statistic_juron.sh $num_p 1 $i $dt
      ((sum+=num_p))
      break
   fi
   ./juron_replace.sh cube_2pop_statistic_juron.sh $num_p 1 $i $dt
   ((sum+=num_p))
done

echo $sum

bhist -u matuschke1
