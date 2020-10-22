#! /usr/bin/env python3

import glob
import os
import json
import subprocess

subprocess.run([
    'find', '.', '-type', 'f', '-iname', '"*.ipynb"', '-not', '-path',
    '"./env*/*"', '-not', '-path', '"./**/env*/*"'
])
