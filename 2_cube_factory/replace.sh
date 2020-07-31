#!/bin/bash

c=0
file=$(cat $1)
for var in "$@"; do
   if (($c == 0)); then
      ((c++))
      continue
   fi
   file=$(sed 's/$'"$c"'/'"$var"'/g' <<<"$file")

   ((c++))
done

echo "$file" >$1.run
echo "$file"

/p/project/cjinm11/Private/matuschke1/run_sbatch.sh $1.run
# rm $1.run
