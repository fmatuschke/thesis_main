#!/bin/bash

echo "source modules"
source jureca_modules.sh
make clean
make VENV=env-jureca install
echo "... done"
