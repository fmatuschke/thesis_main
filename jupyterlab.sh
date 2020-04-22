#!/bin/bash

set -euo pipefail

if [ ! -d "env-$(hostname)" ]; then
   python3 -m venv env-$(hostname)
fi

if [ ! -f "env-$(hostname)/bin/pip3" ]; then
   echo "venv comprimised, rm and reinstall"
   rm -rf env-$(hostname)
   python3 -m venv env-$(hostname)
fi

if ! env-$(hostname)/bin/pip3 freeze | grep jupyterlab= -q; then
   env-$(hostname)/bin/pip3 install jupyterlab
   env-$(hostname)/bin/pip3 install jupyterlab-git
fi

env-$(hostname)/bin/pip3 install -r requirements.txt
# echo "install requirements.txt"
# find . -maxdepth 2 -name "requirements.txt" -exec env-$(hostname)/bin/pip3 install -q -r {} \;
