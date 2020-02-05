import numpy as np
import copy
import h5py
import warnings

from tqdm import tqdm

from fastpli.analysis.images import fom_hsv_black
from fastpli.simulation import optic


def run_simulation_pipeline_n(simpli,
                              label_field,
                              vector_field,
                              tissue_properties,
                              n_repeat,
                              h5f=None,
                              crop_tilt=False,
                              mp_pool=None):
    '''
    copy from _simpli.py
    '''

    if simpli._tilts is None:
        raise ValueError("tilts is not set")
    if simpli._optical_sigma is None:
        raise ValueError("optical_sigma is not set")

    flag_rofl = True
    if np.any(simpli._tilts[:, 1] != np.deg2rad([0, 0, 90, 180, 270])
             ) or simpli._tilts[0, 0] != 0 or np.any(
                 simpli._tilts[1:, 0] != simpli._tilts[1, 0]):
        warnings.warn("Tilts not suitable for ROFL. Skipping analysis")
        flag_rofl = False

    tilting_stack = [None] * len(simpli._tilts)

    simpli._print("Simulate tilts:")
    for t, tilt in enumerate(simpli._tilts):
        theta, phi = tilt[0], tilt[1]
        simpli._print("{}: theta: {} deg, phi: {} deg".format(
            t, round(np.rad2deg(theta), 2), round(np.rad2deg(phi), 2)))

        # images_n = []
        new_images_n = []
        images = simpli.run_simulation(label_field, vector_field,
                                       tissue_properties, theta, phi)

        if crop_tilt:
            delta_voxel = simpli.crop_tilt_voxel()
            images = images[delta_voxel:-1 - delta_voxel,
                            delta_voxel:-1 - delta_voxel, :]

        # apply optic to simulation
        if not simpli._sensor_gain:
            raise ValueError("sensor_gain not set")
        output_org = simpli.apply_optic_resample(images, mp_pool=mp_pool)
        if np.amin(output_org) < 0:
            raise AssertionError("intensity < 0 detected")

        for _ in range(n_repeat):
            if simpli._sensor_gain > 0:
                output = optic.add_noise(output_org, simpli._sensor_gain)
            else:
                output = output_org
            new_images_n.append(output.copy())

        new_images = np.vstack(new_images_n)

        if h5f:
            h5f['simulation/data/' + str(t)] = images
            h5f['simulation/data/' + str(t)].attrs['theta'] = theta
            h5f['simulation/data/' + str(t)].attrs['phi'] = phi
        if h5f:
            h5f['simulation/optic/' + str(t)] = new_images
            h5f['simulation/optic/' + str(t)].attrs['theta'] = theta
            h5f['simulation/optic/' + str(t)].attrs['phi'] = phi

        # calculate modalities
        epa = simpli.apply_epa(new_images)

        if h5f:
            h5f['analysis/epa/' + str(t) + '/transmittance'] = epa[0]
            h5f['analysis/epa/' + str(t) + '/direction'] = np.rad2deg(epa[1])
            h5f['analysis/epa/' + str(t) + '/retardation'] = epa[2]

            h5f['analysis/epa/' + str(t) +
                '/transmittance'].attrs['theta'] = theta
            h5f['analysis/epa/' + str(t) + '/transmittance'].attrs['phi'] = phi
            h5f['analysis/epa/' + str(t) + '/direction'].attrs['theta'] = theta
            h5f['analysis/epa/' + str(t) + '/direction'].attrs['phi'] = phi
            h5f['analysis/epa/' + str(t) +
                '/retardation'].attrs['theta'] = theta
            h5f['analysis/epa/' + str(t) + '/retardation'].attrs['phi'] = phi

        tilting_stack[t] = new_images

    # pseudo mask
    mask = np.sum(label_field, 2) > 0
    mask = simpli.apply_optic_resample(1.0 * mask, mp_pool=mp_pool) > 0.1
    h5f['simulation/optic/mask'] = np.uint8(mask)

    tilting_stack = np.array(tilting_stack)
    while tilting_stack.ndim < 4:
        tilting_stack = np.expand_dims(tilting_stack, axis=-2)

    if flag_rofl:
        simpli._print("Analyse tilts")
        rofl_direction, rofl_incl, rofl_t_rel, (
            rofl_direction_conf, rofl_incl_conf, rofl_t_rel_conf, rofl_func,
            rofl_n_iter) = simpli.apply_rofl(tilting_stack,
                                             mask=None,
                                             mp_pool=mp_pool)
    else:
        rofl_direction = None
        rofl_incl = None
        rofl_t_rel = None

        rofl_direction_conf = None
        rofl_incl_conf = None
        rofl_t_rel_conf = None
        rofl_func = None
        rofl_n_iter = None

    if h5f and flag_rofl:
        h5f['analysis/rofl/direction'] = rofl_direction
        h5f['analysis/rofl/inclination'] = rofl_incl
        h5f['analysis/rofl/t_rel'] = rofl_t_rel

        h5f['analysis/rofl/direction_conf'] = rofl_direction_conf,
        h5f['analysis/rofl/inclination_conf'] = rofl_incl_conf,
        h5f['analysis/rofl/t_rel_conf'] = rofl_t_rel_conf,
        h5f['analysis/rofl/func'] = rofl_func,
        h5f['analysis/rofl/n_iter'] = rofl_n_iter

    if flag_rofl:
        fom = fom_hsv_black(rofl_direction, rofl_incl)
    else:
        fom = None

    return tilting_stack, (rofl_direction, rofl_incl, rofl_t_rel), fom
