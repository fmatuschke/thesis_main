# configfile: "config1.json"

VOXEL_SIZES = [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
# VOXEL_SIZES = [2.5, 5]
LENGTH = VOXEL_SIZES[-1]*10

PSI = "0.5"
OMEGA = list("{:.1f}".format(i) for i in range(10, 100, 10))

rule all:
    threads: 1
    input:
        expand('output/cube_2pop_psi_{psi}_omega_{omega}.solved.h5' + f".simulation_vref_{VOXEL_SIZES[0]}_length_{LENGTH:.0f}.h5", psi = PSI, omega = OMEGA)
        
rule simulation:
    input:
        "input/cube_2pop_psi_{psi}_omega_{omega}.solved.h5",
    output:
        "output/cube_2pop_psi_{psi}_omega_{omega}.solved.h5" + f".simulation_vref_{VOXEL_SIZES[0]}_length_{LENGTH:.0f}.h5"
    threads: 4
    resources: 
        mem_mb=(LENGTH / 0.05)**2 * 60/0.05 * (32 + 32 + 3 * 32) / 8 / 1024**2
    message: "Executing somecommand with {threads} threads on the following files {input}."
    shell:
        "./python3 simulation.py --input {input} --num-threads {threads} "+ f"--voxel-size {' '.join(str(i) for i in VOXEL_SIZES)} --length {LENGTH}"
