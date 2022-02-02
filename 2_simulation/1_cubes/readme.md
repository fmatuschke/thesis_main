# RUN

generate data

```sh
# activate env
./simulation_ime.sh
```

## post

```sh
for p in cube_2pop_135_rc1 cube_2pop_135_rc1_flat cube_2pop_135_rc1_r_0.5 cube_2pop_135_rc1_single cube_2pop_135_rep_rc1_flat cube_2pop_135_rep_rc1_inclined cube_2pop_135_rep_rc1_single; do; python3 simulation_post_0.py -i /data/PLI-Group/felix/data/thesis/2_simulation/1_cubes/output/$p -p 48; done
```

## images

### pre analysis, which radius to choose

```sh
./simulation_mp.sh
python3 generate_tissue.py -i output/cube_2pop_135_rc1 -p 48
```

### single population analysis

```sh
python3 plots_single_pop.py
./plot.sh plots_single_pop.tex output/cube_2pop_135_rc1_single
```

### flat population analysis

```sh
#  change psi value
python3 plots_flat_pop.py
./plot.sh plots_flat_pop.tex output/cube_2pop_135_rc1_flat
```

### inclined population analysis

```sh
#  change psi value
python3 plots_inclined_pop.py
./plot.sh plots_inclined_pop.tex output/cube_2pop_135_rc1_inclined
```

### polar histogramms

```sh
python3 simulation_analysis_hist.py -i output/cube_2pop_135_rc1_r_0.5/
```
