# RUN

generate data

```sh
# activate env
python3 cube_2pop_ime.py

python3 cube_2pop_post_0.py -i output/cube_2pop_... -p 48
python3 cube_2pop_post_1.py -i output/cube_2pop_... -p 48
# python3 cube_2pop_orientation_dist.py -i output/cube_2pop_... -p 48

# rep
python3 cube_2pop_rep.py -o output/cube_2pop_135_rep_rc1 -r 0.5 -n 100000 -p 48 -v 135
python3 cube_2pop_post_rep_0.py -i output/cube_2pop_... -p 48
python3 cube_2pop_post_rep_1.py -i output/cube_2pop_... -p 48


```

## tex files

```sh
# check path inside tex file !
python3 pre_stats_post_0.py -i output/parameter_statistic -p 48

python3 pre_stats_time_evolve.py -i output/parameter_statistic -p 48
./pre_stats_time_evolve.sh output/parameter_statistic

python3 pre_stats_box_plot.py -i output/parameter_statistic
./pre_stats_box_plot.sh output/parameter_statistic
```

df\_ = df.apply(pd.Series.explode).reset_index()
