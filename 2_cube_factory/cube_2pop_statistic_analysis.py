import numpy as np
import itertools
import h5py
import os
import sys
import glob

import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import pandas as pd
import tikzplotlib

import helper.circular
import fastpli.io
import fastpli.tools
import fastpli.objects
import fastpli.analysis

df = pd.read_pickle("output/cube_stat/cube_stat.pkl")
df = df[df.state == "solved"]
df = df[df.omega == 90.0]
df = df[df.psi == 0.5]
df = df[df.fl <= 6]
df['volume'] = df.count_elements.map(
    lambda x: np.sum(x[1:])) / df.count_elements.map(np.sum)
df['frac_num_col_obj'] = df.num_col_obj / df.num_obj

# palette = "husl"
palette = "colorblind"
# colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
# palette = sns.xkcd_palette(colors)

ys = ["time", "overlap", "frac_num_col_obj", "volume", "num_steps"]
# ys = ["time", "overlap", "num_col_obj", "num_obj", "num_steps", "volume"]

# for i, y in enumerate(ys):
#     f, ax = plt.subplots(1, 1, figsize=(15, 15))
#     sns.despine(f, left=True, bottom=True)

# df_ = pd.DataFrame()
# for fl in df.fl.unique():
#     for fr in df.fr.unique():
#         df_[f"fr_{fr}_fl_{fl}"] = df[(df.fl == fl) & (df.fr == fr)][y]
#         print(fl, fr, df[(df.fl == fl) & (df.fr == fr)][y])

# df_.to_csv(f'file_{y}.csv', index=False)

print(df.fr.unique())
print(df.fl.unique())
df[ys + ['fr', 'fl']].to_csv('file.csv', index=False)
