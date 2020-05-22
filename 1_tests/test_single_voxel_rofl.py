import numpy as np

import fastpli.simulation
import fastpli.analysis
import fastpli.tools
import fastpli.objects

import matplotlib.pyplot as plt

print(fastpli.__version__)

np.random.seed(42)

### Setup Simpli for Tissue Generation
simpli = fastpli.simulation.Simpli()
simpli.omp_num_threads = 1
simpli.voxel_size = 60  # in mu meter
simpli.dim = [1, 1, 1]

# single fiber in x-direction
simpli.fiber_bundles = [[np.array([[0, 0, -1e3, 1e3], [0, 0, 1e3, 1e3]])]]

rot = fastpli.tools.rotation.a_on_b([0, 0, 1], [1, 1, 0])
simpli.fiber_bundles = fastpli.objects.fiber_bundles.Rotate(
    simpli.fiber_bundles, rot)

simpli.fiber_bundles_properties = [[(0.333, -0.001, 10, 'p')]]
### Generate Tissue
print("Run Generation:")
label_field, vec_field, tissue_properties = simpli.generate_tissue()

### Simulate PLI Measurement ###
simpli.filter_rotations = np.deg2rad([0, 30, 60, 90, 120, 150])
simpli.light_intensity = 26000  # a.u.
simpli.interpolate = False
simpli.wavelength = 525  # in nm
TILTS = [(0, 0), (5.5, 0), (5.5, 90), (5.5, 180), (5.5, 270)]

tilting_stack = [None] * len(TILTS)
print("Run Simulation:")
for t, (theta, phi) in enumerate(TILTS):
    print(theta, phi)
    images = simpli.run_simulation(label_field, vec_field, tissue_properties,
                                   np.deg2rad(theta), np.deg2rad(phi))

    # calculate modalities
    trans, direct, retard = simpli.apply_epa(images)
    tilting_stack[t] = images

    plt.plot(np.rad2deg(simpli.filter_rotations), images.flatten())
    print(images.flatten())

print("Run ROFL analysis:")
rofl_direction, rofl_incl, rofl_t_rel, _ = simpli.apply_rofl(
    tilting_stack, tilt_angle=np.deg2rad(TILTS[-1][0]))

print("FB:", simpli.fiber_bundles[0][0])
print("EPA:",
      trans.flatten()[0], np.rad2deg(direct.flatten()[0]),
      retard.flatten()[0])
print(
    "t_rel:", 4 * simpli.voxel_size *
    abs(simpli.fiber_bundles_properties[0][0][1]) / (simpli.wavelength / 1e3))
print("ROFL:", np.rad2deg(rofl_direction.flatten()[0]),
      rofl_incl.flatten()[0],
      rofl_t_rel.flatten()[0])

plt.show()
