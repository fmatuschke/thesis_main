import numpy as np
import h5py
import os
import sys

import fastpli.simulation
import fastpli.analysis
import fastpli.objects
import fastpli.tools
import fastpli.io

# reproducability
np.random.seed(42)


def simulation(input,
               output,
               voxel_sizes,
               length,
               thickness,
               rot_f0=0,
               rot_f1=0,
               num_threads=1):

    if os.path.isfile(output):
        print("output already exists:", output)
        return

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with h5py.File(output, 'w-') as h5f:
        fiber_bundles = fastpli.io.fiber_bundles.load(input, '/')

        if rot_f0 or rot_f1:
            rot_inc = fastpli.tools.rotation.y(-np.deg2rad(rot_f0))
            rot_phi = fastpli.tools.rotation.x(np.deg2rad(rot_f1))
            fiber_bundles = fastpli.objects.fiber_bundles.Rotate(
                fiber_bundles, np.dot(rot_inc, rot_phi))

        with open(os.path.abspath(__file__), 'r') as script:
            h5f.attrs['script'] = script.read()

        for voxel_size in voxel_sizes:
            for dn, model in [(-0.001, 'p'), (0.002, 'r')]:
                dset = h5f.create_group(str(voxel_size) + '/' + model)

                # Setup Simpli
                simpli = fastpli.simulation.Simpli()
                simpli.omp_num_threads = num_threads
                simpli.voxel_size = voxel_size
                simpli.resolution = voxel_size
                simpli.filter_rotations = np.deg2rad([0, 30, 60, 90, 120, 150])
                simpli.interpolate = True
                simpli.untilt_sensor_view = True
                simpli.wavelength = 525  # in nm

                simpli.set_voi(-0.5 * np.array([length, length, thickness]),
                               0.5 * np.array([length, length, thickness]))

                # simpli.tilts = np.deg2rad(
                #     np.array([(0, 0), (5.5, 0), (5.5, 90), (5.5, 180),
                #               (5.5, 270)]))
                simpli.tilts = np.deg2rad(np.array([(0, 0)]))
                simpli.add_crop_tilt_halo()

                if simpli.memory_usage() > 24 * 1e3:
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
