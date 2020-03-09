#!/bin/bash

while read line; do    
   if [[ "$line" =~ "SBATCH --nodes=" ]]; then
      NODES=`echo $line | tr -d -c 0-9`
   fi
   if [[ "$line" =~ "SBATCH --ntasks=" ]]; then
      NTASKS=`echo $line | tr -d -c 0-9`
   fi
   if [[ "$line" =~ "SBATCH --time=" ]]; then
      TIME=`echo $line | tr -d -c 0-9:`
   fi
done < $*

# echo $TIME

HOURS=`echo $TIME | cut -d: -f1`
MINUTES=`echo $TIME | cut -d: -f2`
SECONDS=`echo $TIME | cut -d: -f3`

COREH=`bc <<< "scale=2; $NODES * 24.0 * $HOURS + $MINUTES / 60.0 + $SECONDS / 60.0 / 60.0"`

echo "Process allocates $COREH CoreH"

if (( $(echo "$COREH > 1000" | bc -l) )); then
   read -p "Are you sure? (y/n) " yn
   if [[ $yn != "yes" && $yn != "y" ]]; then
      exit
   fi
fi

sbatch $*
