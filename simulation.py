import numpy as np
import argparse
import h5py
import os

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--num-threads',
                    default=1,
                    type=int,
                    required=False,
                    help='number of omp threads (4 is a good choice)')
parser.add_argument('--input',
                    type=str,
                    required=True,
                    help='input h5 file of model')
parser.add_argument('--voxel-size',
                    nargs='+',
                    type=float,
                    required=True,
                    help='values of voxel sizes')
parser.add_argument('--length',
                    type=float,
                    required=True,
                    help='length of model')
args = parser.parse_args()

NUM_THREADS = args.num_threads
INPUT_FILE = args.input
LENGTH = args.length
VOXEL_SIZES = args.voxel_size

# reproducability
np.random.seed(42)

OUTPUT_NAME = 'output/' + os.path.basename(
    INPUT_FILE) + f".simulation_vref_{VOXEL_SIZES[0]}_length_{LENGTH:.0f}"

if not os.path.isfile(OUTPUT_NAME + '.h5'):
    print("simulate data")

    import fastpli.simulation
    import fastpli.analysis
    import fastpli.objects
    import fastpli.tools
    import fastpli.io

    # PARAMETER
    THICKNESS = 60
    FIBER_INCLINATION = np.linspace(0, 90, 10, True)

    fiber_bundles = fastpli.io.fiber_bundles.load(INPUT_FILE, '/')

    os.makedirs('output', exist_ok=True)

    with h5py.File(OUTPUT_NAME + '.h5', 'w-') as h5f:
        with open(os.path.abspath(__file__), 'r') as script:
            h5f.attrs['script'] = script.read()

        for voxel_size in tqdm(VOXEL_SIZES):
            for dn, model in [(-0.001, 'p'), (0.002, 'r')]:
                dset = h5f.create_group(str(voxel_size) + '/' + model)

                # Setup Simpli
                simpli = fastpli.simulation.Simpli()
                simpli.omp_num_threads = NUM_THREADS
                simpli.voxel_size = voxel_size
                simpli.resolution = voxel_size
                simpli.filter_rotations = np.deg2rad([0, 30, 60, 90, 120, 150])
                simpli.interpolate = True
                simpli.untilt_sensor_view = True
                simpli.wavelength = 525  # in nm

                simpli.set_voi(-0.5 * np.array([LENGTH, LENGTH, THICKNESS]),
                               0.5 * np.array([LENGTH, LENGTH, THICKNESS]))

                # simpli.tilts = np.deg2rad(
                #     np.array([(0, 0), (5.5, 0), (5.5, 90), (5.5, 180),
                #               (5.5, 270)]))
                simpli.tilts = np.deg2rad(np.array([(0, 0)]))
                simpli.add_crop_tilt_halo()

                # print(simpli.dim)
                if simpli.memory_usage() > 1e3:
                    print(str(round(simpli.memory_usage(), 2)) + 'MB')

                simpli.fiber_bundles = fiber_bundles
                simpli.fiber_bundles_properties = [[(0.75, 0, 0, 'b'),
                                                    (1.0, dn, 0, model)]
                                                  ] * len(fiber_bundles)
                label_field, vector_field, tissue_properties = simpli.run_tissue_pipeline(
                    h5f=dset, save=["label_field"])

                # Simulate PLI Measurement
                simpli.light_intensity = 1  # a.u.
                simpli.save_parameter_h5(h5f=dset)
                for t, tilt in enumerate(simpli._tilts):
                    theta, phi = tilt[0], tilt[1]
                    images = simpli.run_simulation(label_field, vector_field,
                                                   tissue_properties, theta,
                                                   phi)

                    images = simpli.rm_crop_tilt_halo(images)

                    dset['simulation/data/' + str(t)] = images
                    dset['simulation/data/' + str(t)].attrs['theta'] = theta
                    dset['simulation/data/' + str(t)].attrs['phi'] = phi

                del label_field
                del vector_field
                del simpli
