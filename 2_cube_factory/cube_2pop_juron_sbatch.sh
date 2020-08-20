#!/bin/bash

set -e

for r in 10 5 2 1 0.5; do
   N=92
   num_p=20
   sum=0
   for ((i=0; i<$N; i=i+$num_p))
   do
      if [ $(($i+$num_p)) -gt $(($N)) ]
      then
         num_p=$(($N-$i))
         ./replace_juron.sh cube_2pop_juron.sh $num_p $i $r cube_2pop_0
         ((sum+=num_p))
         break
      fi
      ./replace_juron.sh cube_2pop_juron.sh $num_p $i $r cube_2pop_0
      ((sum+=num_p))
      echo ""
   done
   echo ""
   echo $sum
   echo ""
done
bhist -u matuschke1
