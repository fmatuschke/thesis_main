#!/bin/bash

set -euo pipefail

if [ ! -d "env-$(hostname)" ]; then
   python3 -m venv env-$(hostname)
   pip3 install -U pip
fi

source env-$(hostname)/bin/activate

if ! pip3 freeze | grep jupyterlab= -q; then
   # pip3 install jupyterlab
   # pip3 install jupyterlab-git
   pip3 install jupyter -q
   pip3 install jupyterthemes -q
   pip3 install jupyter_contrib_nbextensions -q
   jupyter contrib nbextension install --user
   jt -t onedork -cellw 99% -T -nf ptsans -lineh 125
   cp custom.css ~/.jupyter/custom/custom.css
fi
