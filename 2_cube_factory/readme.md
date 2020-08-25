# readme

- \*.sh : run all
- \*.py : erzeuge daten
- \*.ipynb : jupyter notebook zum entwickeln
- \*\_jureca.\* : jureca versionen
- \*\_post\_\{i\}.py : i-ter post processing schritt
- \*\_analysis\_\{name\}.py : analyse der \{name\} ergebnisse
- \*\_analysis\_\{name\}.tex : texen der ergebnisse
- \*\_analysis\_\{name\}.sh : script zur erstellung einer pdf aus tikz einzel bilder

## run

```sh
./cube_2pop_post_0.py -i output/cube_2pop_0 -p 16
./cube_2pop_orientation_hist.py -i output/cube_2pop_0 -p 16
./cube_2pop_images.py -i output/cube_2pop_0 -v 60 -p 4

./cube_2pop_orientation_hist.sh
```
