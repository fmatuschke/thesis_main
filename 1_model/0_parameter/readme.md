# RUN

generate data

```sh
# activate env
./pre_stats.py -o output/parameter_statistic -p 48
```

## images

```sh
python3 pre_stats_post_0.py -i output/... -p 48

python3 pre_stats_time_evolve.py -i output/... -p 48
./plot.sh pre_stats_time_evolve.tex output/...

python3 pre_stats_box_plot.py -i output/...
./plot.sh pre_stats_box_plot.tex
```

df\_ = df.apply(pd.Series.explode).reset_index()
