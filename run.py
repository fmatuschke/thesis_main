import numpy as np
import multiprocessing
import os
import glob
import argparse
import logging

from tqdm import tqdm
from simulation import simulation
from optic_and_epa import optic_and_epa

OMEGAS = [0, 30, 60, 90]
VOXEL_SIZES = [0.025, 0.05, 0.1, 0.25, 0.75, 1.25]
RESOLUTIONS = [1.25, 2.5]
LENGTH = VOXEL_SIZES[-1] * 10

# logger = logging.getLogger("rank[%i]" % comm.rank)
# logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--input',
                    type=str,
                    required=True,
                    nargs='+',
                    help='input h5 files')
parser.add_argument('--output', type=str, required=True, help='input h5 files')
parser.add_argument('-n',
                    '--num-cpus',
                    type=int,
                    required=True,
                    help='number of max cpus to use')
parser.add_argument('-t',
                    '--num-threads-per-process',
                    type=int,
                    required=True,
                    help='number of threads for each process')
parser.add_argument('--all', action='store_true')
parser.add_argument('--simulation', action='store_true')
parser.add_argument('--optic', action='store_true')
args = parser.parse_args()


def _rot_pop2_factory(omega, delta_omega):
    import fastpli.tools
    import fastpli.objects

    inc_list = np.arange(0, 90 + 1e-9, delta_omega)

    n_rot = int(
        round(2 * np.pi / np.deg2rad(delta_omega) * np.sin(np.deg2rad(omega))))

    if n_rot == 0:
        fiber_phi_rotations = [0]
    else:
        n_rot = max(n_rot, 8)
        fiber_phi_rotations = np.linspace(0, 360, n_rot, False)

    for f0_inc in inc_list:
        for f1_rot in fiber_phi_rotations:
            yield f0_inc, f1_rot


def _simulation(parameter):
    file, f0, f1 = parameter
    file_name = os.path.basename(file)
    file_name = file_name.rpartition(".solved")[0]
    output = f"{args.output}/{file_name}_vref_{VOXEL_SIZES[0]}_length_{LENGTH}_f0_{f0}_f1_{f1}.simulation.h5"

    if os.path.isfile(output):
        return output

    simulation(input=file,
               output=output,
               voxel_sizes=VOXEL_SIZES,
               length=LENGTH,
               thickness=60,
               rot_f0=f0,
               rot_f1=f1,
               num_threads=args.num_threads_per_process)

    return output


def _optic_and_epa(files):
    files.sort()
    file_name = files[0]
    file_name = file_name.rpartition(".simulation")[0] + '.analysis.pkl'
    optic_and_epa(input=files, output=file_name, resolution=RESOLUTIONS)


if __name__ == '__main__':

    # MP
    num_cpus = min(multiprocessing.cpu_count(), args.num_cpus)
    pool = multiprocessing.Pool(processes=num_cpus //
                                args.num_threads_per_process)

    # Simulation
    # if args.all or args.simulation:
    print("--------SIMULATION--------")
    # files = glob.glob(f"input/*.solved.h5")
    files = args.input
    parameter = []
    for file in files:
        omega = float(file.split("_omega_")[-1].split(".solved")[0])
        if not omega in OMEGAS:
            continue
        for f0, f1 in _rot_pop2_factory(omega, OMEGAS[1] - OMEGAS[0]):
            parameter.append((file, f0, f1))

    print(f"len files: {len(files)}")
    print(f"num parameters: {len(parameter)}")
    files = [
        i for i in tqdm(pool.imap(_simulation, parameter), total=len(parameter))
    ]

    # Optic and Analysis
    if args.all or args.optic:
        print("-----------O&A-----------")
        _optic_and_epa(files)
