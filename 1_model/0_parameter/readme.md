# RUN

generate data

```sh
# activate env
./pre_stats.py -o output/parameter_statistic -p 48
```

## images

```sh
# check path inside tex file !
python3 pre_stats_post_0.py -i output/parameter_statistic -p 48

python3 pre_stats_time_evolve.py -i output/parameter_statistic -p 48
./pre_stats_time_evolve.sh

python3 pre_stats_box_plot.py -i output/parameter_statistic
./pre_stats_box_plot.sh
```

df\_ = df.apply(pd.Series.explode).reset_index()
