# configfile: "config1.json"

VOXEL_SIZES = [0.01, 0.025, 0.05, 0.1, 0.25, 0.75, 1.25]
RESOLUTIONS = [1.25, 2.5]
LENGTH = VOXEL_SIZES[-1]*10

PSI = "0.5"
OMEGA = list("{:.1f}".format(i) for i in range(10, 100, 10))

# print(expand("output/cube_2pop_{parameter}.solved.h5"))

rule all:
    input:
        f"output/cube_2pop_vref_{VOXEL_SIZES[0]}_length_{LENGTH:.0f}.pkl"
        
rule simulation:
    input:
        "input/cube_2pop_psi_{psi}_omega_{omega}.solved.h5",
    output:
        "output/cube_2pop_psi_{psi}_omega_{omega}.solved.h5" +
            f".simulation_vref_{VOXEL_SIZES[0]}_length_{LENGTH:.0f}.h5"
    threads: 4
    resources: 
        mem_mb= int((LENGTH / 0.05)**2 * 60/0.05 * (32 + 32 + 3 * 32) / 8 / 1024**2)
    message: "Executing somecommand with {threads} threads on the following files {input}."
    shell:
        "./python3 simulation.py --input {input} --num-threads {threads} " +
            f"--voxel-size {' '.join(str(i) for i in VOXEL_SIZES)} --length {LENGTH}"

rule analysis:
    input:
        "output/cube_2pop_psi_{psi}_omega_{omega}.solved.h5" +
            f".simulation_vref_{VOXEL_SIZES[0]}_length_{LENGTH:.0f}.h5"
        # rules.simulation.output
        # expand("output/cube_2pop_psi_{psi}_omega_{omega}.solved.h5.simulation_vref_{vref}_length_{length}.h5",
        #     psi=PSI, omega=OMEGA, vref=VOXEL_SIZES[0], length=LENGTH)
    output:
        f"output/cube_2pop_vref_{VOXEL_SIZES[0]}_length_{LENGTH:.0f}.pkl"
    shell:
        "./python3 analysis.py --input {input} --output {output} " +
            f"--resolution {' '.join(str(i) for i in RESOLUTIONS)}"
