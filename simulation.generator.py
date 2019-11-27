import yep

import fastpli.simulation
import fastpli.analysis
import fastpli.io

import numpy as np
import h5py
import sys
import os

import time

FILE_NAME = os.path.abspath(__file__)
FILE_PATH = os.path.dirname(FILE_NAME)
FILE_BASE = os.path.basename(FILE_NAME)
FILE_OUTPUT = os.path.join(FILE_PATH, 'output')
os.makedirs(FILE_OUTPUT, exist_ok=True)

yep.start(
    os.path.join(FILE_OUTPUT,
                 FILE_BASE + '.' + fastpli.__git__hash__ + '.prof'))

np.random.seed(42)

with h5py.File(
        FILE_OUTPUT + '/' + FILE_BASE + '.' + fastpli.__git__hash__ + '.h5',
        'w') as h5f:

    # save script
    with open(os.path.abspath(__file__), 'r') as f:
        try:
            from pip._internal.operations import freeze
        except ImportError:
            from pip.operations import freeze
        h5f['script'] = f.read()
        h5f['pip_freeze'] = "\n".join(freeze.freeze())

    # Setup Simpli for Tissue Generation
    simpli = fastpli.simulation.Simpli()
    simpli.omp_num_threads = 2
    simpli.voxel_size = 0.2  # in mu meter
    simpli.set_voi([-60] * 3, [60] * 3)  # in mu meter
    simpli.fiber_bundles = fastpli.io.fiber.load(
        os.path.join(FILE_PATH, 'fastpli/examples/cube.dat'))
    simpli.fiber_bundles_properties = [[(0.333, -0.004, 10, 'p'),
                                        (0.666, 0, 5, 'b'),
                                        (1.0, 0.004, 1, 'r')]]

    print('VOI:', simpli.get_voi())
    print('Memory:', str(round(simpli.memory_usage('MB'), 2)) + ' MB')

    # Generate Tissue
    print("Run Generation:")
    t0 = time.time()
    label_field, vec_field, tissue_properties = simpli.generate_tissue()
    print(time.time() - t0)

yep.stop()

os.system("google-pprof --callgrind " + sys.executable + " " + FILE_OUTPUT +
          '/' + FILE_BASE + '.' + fastpli.__git__hash__ + ".prof > " +
          FILE_OUTPUT + '/' + FILE_BASE + '.' + fastpli.__git__hash__ +
          ".callgrind")
