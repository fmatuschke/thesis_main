# Allgemein

`optic.py` and `optic.ipynb` ist nur zum testen

# USAF / Optic / resolution

```sh
python3 resolution.ipy
```

# gain factor / noise

```sh
./sensor_gain.py
```

# absorption coefficient / birefringence

```sh
python3 measure_vervet.ipy
./plot.sh birefringence.tex output/bf_rc1 --single
```

# voxel size

```sh
# simulation
python3 voxel_size.py -i ../../1_model/1_cubes/output/cube_2pop_135_rc1 -o output/vs_135_0.01_6_25_rc1 -p 10 -t 1 -m 6 -n 25

# analyse
python3 voxel_size_post_0.py -i output/vs_135_0.01_6_25_rc1 -p 16
python3 voxel_size_post_1.py -i output/vs_135_0.01_6_25_rc1 -p 16

# generate plot csv data
python3 voxel_size_plots.py -i output/vs_135_0.01_6_25_rc1

# plots
# _0 without noise, _1 with noise
./plot.sh voxel_size_plots_data_r05.tex output/vs_135_0.01_6_25_vervet_r_rc1
./plot.sh voxel_size_plots_data.tex output/vs_135_0.01_6_25_vervet_r_rc1
```

# fiber radii
