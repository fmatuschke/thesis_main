#!/bin/bash
set -e

echo "source modules"
source jureca_modules.sh

mkdir ${PWD}/pip

pip3 uninstall fastpli
(
   cd fastpli
   make clean
   make fastpli
)
pip3 install --install-option="--prefix=${PWD}/pip" fastpli/.
pip3 install --install-option="--prefix=${PWD}/pip" -r requirements.txt
echo "... done"
