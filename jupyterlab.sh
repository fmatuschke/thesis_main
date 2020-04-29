#!/bin/bash

set -euo pipefail

if [ -f "/usr/bin/python3.8" ]; then
   python3='/usr/bin/python3.8'
else
   python3=python3
fi

if [ ! -d "env-$(hostname)" ]; then
   $python3 -m venv env-$(hostname)
fi

source env-$(hostname)/bin/activate

if ! pip3 freeze | grep jupyterlab= -q; then
   pip3 install jupyterlab
   pip3 install jupyterlab-git
   pip3 install jupyterthemes
   pip3 install jupyter_contrib_nbextensions
   jupyter contrib nbextension install --user
   jt -t onedork -cellw 99% -T -nf ptsans -lineh 125
fi

#pip3 install -r requirements.txt


# echo "install requirements.txt"
# find . -maxdepth 2 -name "requirements.txt" -exec pip3 install -q -r {} \;
