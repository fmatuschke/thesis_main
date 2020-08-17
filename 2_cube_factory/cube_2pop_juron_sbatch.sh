#!/bin/bash

set -e

N=92
n=20
num_p=80
sum=0

# for r v in {210,1,105,0.5}
# do
for ((i=0; i<$N; i=i+$num_p))
do
   if [ $(($i+$num_p)) -gt $(($N-1)) ]
   then
      num_p=$(($N-$i+1))
      ./juron_replace.sh parameter_statistic_juron.sh $num_p $i 1 210
      ((sum+=num_p))
      break
   fi
   ./juron_replace.sh parameter_statistic_juron.sh $num_p $i 1 210
   ((sum+=num_p))
done
# done

for ((i=0; i<$N; i=i+$num_p))
do
   if [ $(($i+$num_p)) -gt $(($N-1)) ]
   then
      num_p=$(($N-$i+1))
      ./juron_replace.sh parameter_statistic_juron.sh $num_p $i 0.5 105
      ((sum+=num_p))
      break
   fi
   ./juron_replace.sh parameter_statistic_juron.sh $num_p $i 0.5 105
   ((sum+=num_p))
done

echo $sum

bhist -u matuschke1
