#!/bin/bash
set -e

if [ "$#" -lt 2 ]; then
   echo "Wrong number of arguments supplied"
   exit
fi

if [ "$#" -gt 3 ]; then
   echo "Wrong number of arguments supplied"
   exit
fi

if [ ! -f "$1" ]; then
   echo "not a file: $1"
   exit
fi

NAME=$(echo "$1" | cut -f 1 -d '.')
APPEND="$(echo $2 | sed 's?/?_?g')"
NAMEOUT="${NAME}_${APPEND}"

mkdir -p output/tmp
mkdir -p output/tikz
cd output/tmp

cp ../../${NAME}.tex ${NAMEOUT}.tex
sed -i 's?__PATH__?'"$2"'?g' ${NAMEOUT}.tex
lualatex -interaction=nonstopmode -halt-on-error ${NAMEOUT}.tex
lualatex -interaction=nonstopmode -halt-on-error ${NAMEOUT}.tex
mv ${NAMEOUT}.pdf ../tikz/${NAMEOUT}.pdf
xdg-open ../tikz/${NAMEOUT}.pdf &>/dev/null 2>&1
