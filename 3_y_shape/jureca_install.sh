#!/bin/bash

echo "source modules"
source jureca_modules.sh
make HOST=jureca install

echo "... done"
