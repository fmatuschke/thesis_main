# RUN

generate data

```sh
# activate env
./simulation_ime.sh
```

## images

### pre analysis, which radius to choose

```sh
./simulation_mp.sh
python3 generate_tissue.py -i output/cube_2pop_135_rc1 -p 48
```

## ### single population analysis

```sh
python3 plots_single_pop.py
```

## ### flat population analysis

```sh
#  change psi value
python3 plots_flat_pop.py
```
