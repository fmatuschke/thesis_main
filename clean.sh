#!/bin/bash

if [ -f env-$(hostname)/bin/activate ]; then
   source env-$(hostname)/bin/activate
elif [ -f env/bin/activate ]; then
   source env/bin/activate
else
   echo 'no env detected'
fi

find . -type f -iname "*.ipynb" -not -path "./env*/*" -not -path "./**/env*/*" -exec python3 -m nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {} \;
