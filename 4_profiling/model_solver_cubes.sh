#!/bin/bash
set -e

prog() {
    local w=$2 p=$1;  shift
    # create a string of spaces, then change them to dots
    printf -v dots "%*s" "$(( $p*$w/100 ))" ""; dots=${dots// /.};
    # print those dots on a fixed-width space plus the percentage etc. 
    printf "\r\e[K|%-*s| %3d %% %s" "$w" "$dots" "$p" "$*"; 
}

_start=0
((_end=8*10))

mkdir -p output
rm output/run.log
echo "p,n,tn,tm">>output/run.log
for p in {1,2,3,4,5,6,7,8}; do
   for n in {1..10}; do
      values=($( \
       python3 model_solver_cubes.py \
       -o output/run_$p_$n \
       -n 100 \
       -m 100 \
       -r 1 \
       -v 60 \
       -p $p \
       --psi 1 \
       --omega 0 \
       ))
      tn=${values[0]}
      tm=${values[1]}
      echo "$p,$n,$tn,$tm">>output/run.log
      let _start=_start+1
      prog ${_start} ${_end}
   done
done
