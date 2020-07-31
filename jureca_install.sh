#!/bin/bash
set -ev

source jureca_modules.sh

if [ -d "env" ]; then
   rm -rf env
fi

(
   cd fastpli
   make clean
   make fastpli
)

python3 -m venv env
source env/bin/activate
#pip3 install --upgrade pip
pip3 install fastpli/.
pip3 install -r requirements.txt
pip3 install 0_core/.

echo "... done"
