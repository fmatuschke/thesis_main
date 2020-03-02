import multiprocessing
import subprocess
import os
import glob
import argparse
import logging

VOXEL_SIZES = [0.025, 0.05, 0.1, 0.25, 0.75, 1.25]
RESOLUTIONS = [1.25, 2.5]
LENGTH = VOXEL_SIZES[-1] * 10

# logger = logging.getLogger("rank[%i]" % comm.rank)
# logger.setLevel(logging.DEBUG)


def list2txt(input):
    return ' '.join(str(i) for i in input)


def simulation(file, threads):
    file_name = os.path.basename(file)
    file_name = file_name.rpartition(".solved")[0]
    output = f"output/{file_name}_vref_{VOXEL_SIZES[0]}_length_{LENGTH}.simulation.h5"
    subprocess.call([
        "./python3", "simulation.py", f"--input {file}", f"--output {output}",
        f"--num-threads {threads}",
        f"--voxel-size {' '.join(str(i) for i in VOXEL_SIZES)}",
        f"--length {LENGTH}"
    ])
    return output


def optic_and_epa(files):
    # file_name = os.path.basename(files[0])
    files.sort()
    file_name = files[0]
    file_name = file_name.rpartition(".simulation")[0] + '.analysis.pkl'
    subprocess.call([
        "./python3", "optic_and_epa.py", f"--input {list2txt(files)}",
        f"--output {file_name}", f"--resolution {list2txt(RESOLUTIONS)}"
    ])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    # MP
    num_cpus = min(multiprocessing.cpu_count(), args.num_cpus)
    pool = multiprocessing.Pool(processes=num_cpus //
                                args.num_threads_per_process)

    # Simulation
    print("--------SIMULATION--------")
    files = glob.glob(f"input/*psi_0.5_*.solved.h5")
    parameter = [(file, args.num_threads_per_process)
                 for file in glob.glob(f"input/*psi_0.5_*.h5")]
    files = pool.starmap(simulation, parameter)

    # Optic and Analysis
    print("-----------O&A-----------")
    optic_and_epa(files)
