import fastpli.model.sandbox
import fastpli.io
import fastpli.tools
import fastpli.objects
import fastpli.model.solver

import time
import numpy as np
import h5py
from tqdm import tqdm, trange
import os

fbs = fastpli.io.fiber.load('output/models/y_shape_hom.0.solved.h5', '/')
fastpli.io.fiber.save('output/models/y_shape_hom.0.solved.dat', fbs)
