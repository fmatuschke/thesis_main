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
   if [ $(($i+$num_p)) -gt $(($N)) ]
   then
      num_p=$(($N-$i))
      ./juron_replace.sh cube_2pop_juron.sh $num_p $i 1 210
      ((sum+=num_p))
      break
   fi
   ./juron_replace.sh cube_2pop_juron.sh $num_p $i 1 210
   ((sum+=num_p))
   echo ""
done
# done

echo ""
echo $sum
echo ""

N=92
n=20
num_p=80
sum=0

for ((i=0; i<$N; i=i+$num_p))
do
   if [ $(($i+$num_p)) -gt $(($N)) ]
   then
      num_p=$(($N-$i))
      ./juron_replace.sh cube_2pop_juron.sh $num_p $i 0.5 105
      ((sum+=num_p))
      break
   fi
   ./juron_replace.sh cube_2pop_juron.sh $num_p $i 0.5 105
   ((sum+=num_p))
   echo ""
done

echo ""
echo $sum
echo ""

bhist -u matuschke1
