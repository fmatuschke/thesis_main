import numpy as np
import h5py
import glob
import os

import fastpli.io


def _pip_freeze():
    try:
        from pip._internal.operations import freeze
    except ImportError:
        from pip.operations import freeze
    return "\n".join(freeze.freeze())


def save_h5_fibers(h5_name,
                   fiber_bundles,
                   file_script,
                   solver_dict=None,
                   solver_step=None,
                   solver_collisions=None,
                   solver_overlap=None):

    fastpli.io.fiber.save(h5_name, fiber_bundles, '/fiber_bundles', 'w-')
    with h5py.File(h5_name, 'r+') as h5f:
        h5f['/fiber_bundles'].attrs['version'] = fastpli.__version__
        h5f['/fiber_bundles'].attrs['pip_freeze'] = _pip_freeze()
        h5f['/fiber_bundles'].attrs['script'] = open(
            os.path.abspath(file_script), 'r').read()

        if solver_dict:
            h5f['/fiber_bundles'].attrs['solver'] = str(solver_dict)
        if solver_step:
            h5f['/fiber_bundles'].attrs['solver.steps'] = solver_step
        if solver_collisions:
            h5f['/fiber_bundles'].attrs[
                'solver.num_col_obj'] = solver_collisions
        if solver_overlap:
            h5f['/fiber_bundles'].attrs['solver.steps'] = solver_overlap


def version_file_name(file_name):

    file_path = os.path.dirname(file_name)
    file_name = os.path.basename(file_name)
    files = glob.glob(file_name)

    print(file_name)

    def in_list(i, file):
        for f in files:
            print(f)
            print(file_name + ".v{}".format(i))
            print()
            if file_name + ".v{}".format(i) in f:
                return True
        return False

    i = 0
    while in_list(i, files):
        i += 1

    return os.path.join(file_path, file_name + ".v{}".format(i))
