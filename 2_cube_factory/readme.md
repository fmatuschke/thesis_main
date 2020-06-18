# readme

- \*.sh : run all
- \*.py : erzeuge daten
- \*.ipynb : jupyter notebook zum entwickeln
- \*\_jureca.\* : jureca versionen
- *\_post\_\{i\}.py : i-ter post processing schritt
- \*\_analysis\_\{name\}.py : analyse der \{name\} ergebnisse
- \*\_analysis\_\{name\}.tex : texen der ergebnisse
- \*\_analysis\_\{name\}.sh : script zur erstellung einer pdf aus tikz einzel bilder

## cube_2pop

erstellung der *cube_2pop* modelle

jureca:

```sh
sbatch cube_2pop_jureca.sh
```

```sh
python3 cube_2pop_post_0.py -i ../data/models/1_rnd_seed/*.h5 -o output/models/1_rnd_seed/ -p 32 # 25min, 20Gb
```

## cube_2pop_statistic

analyse, welche parameter für *cube_2pop* verwendet werden sollen

## simulation

simulation der *cube_2pop* modelle

### spheres

schiller metrik zwischen simulations orientierungs ergebnissen und ground truth
