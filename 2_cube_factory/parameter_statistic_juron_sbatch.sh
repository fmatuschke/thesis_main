#!/bin/bash

set -e

N=800
num_p=40
sum=0
dt=1

for ((i=0; i<$N; i=i+$num_p))
do
   if [ $i -gt 200 ]; then dt=2; fi
   if [ $i -gt 400 ]; then dt=24; fi

   if [ $(($i+$num_p)) -gt $(($N-1)) ]
   then
      num_p=$(($N-$i+1))
      ./juron_replace.sh parameter_statistic_juron.sh $num_p 1 $i $dt
      ((sum+=num_p))
      break
   fi
   ./juron_replace.sh parameter_statistic_juron.sh $num_p 1 $i $dt
   ((sum+=num_p))
done

echo $sum

bhist -u matuschke1
