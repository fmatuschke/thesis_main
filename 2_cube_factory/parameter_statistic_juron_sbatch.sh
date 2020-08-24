#!/bin/bash

set -e

N=800
num_p=20
sum=0
dt=1

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
fi

for ((i=0; i<$N; i=i+$num_p))
do
   if [ $i -gt $((N/n*0)) ]; then dt=1; fi
   if [ $i -gt $((N/n*3)) ]; then dt=6; fi
   if [ $i -gt $((N/n*4)) ]; then dt=24; fi

   if [ $(($i+$num_p)) -gt $(($N-1)) ]
   then
      num_p=$(($N-$i+1))
      ./replace_juron.sh parameter_statistic_juron.sh $num_p 1 $i $dt $1
      ((sum+=num_p))
      break
   fi
   ./replace_juron.sh parameter_statistic_juron.sh $num_p 1 $i $dt $1
   ((sum+=num_p))
done

echo $sum

bhist -u matuschke1
