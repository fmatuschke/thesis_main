#!/bin/bash

set -euxo pipefail

if [ ! -d "env-$(hostname)" ]; then
   conda create --prefix ./env-ime263
   conda install jupyterlab
fi

# if [ ! -d "env-$(hostname)" ]; then
#    python3 -m venv env-$(hostname)
# fi

# if [ ! -f "env-$(hostname)/bin/pip3" ]; then
#    echo "venv comprimised, rm and reinstall"
#    rm -rf env-$(hostname)
#    python3 -m venv env-$(hostname)
# fi

# if ! env-$(hostname)/bin/pip3 freeze | grep jupyterlab= -q; then
#    env-$(hostname)/bin/pip3 install jupyterlab
#    env-$(hostname)/bin/pip3 install jupyterlab-git
# fi
