#!/usr/bin/env python3

import numpy as np
import os
import sys
import glob
import argparse

import pandas as pd

import helper.circular
import fastpli.io
import fastpli.tools
import fastpli.objects
import fastpli.analysis

parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="Path of file.")
args = parser.parse_args()
os.makedirs(os.path.join(args.input, "boxplot"), exist_ok=True)

df = pd.read_pickle(os.path.join(args.input, "parameter_statistic.pkl"))
df = df[df.state == "solved"]
# df = df[df.omega == 90.0]
# df = df[df.psi == 0.5]
# df = df[df.fl <= 6]

df['volume'] = df.count_elements.map(
    lambda x: np.sum(x[1:])) / df.count_elements.map(np.sum)
df['frac_num_col_obj'] = df.num_col_obj / df.num_obj
df['frac_overlap'] = df.overlap / df.num_col_obj
df = df.replace([np.inf, np.nan], 0)

for r in df.r.unique():
    df_ = pd.DataFrame()
    for omega in df.omega.unique():
        for psi in df[df.omega == omega].psi.unique():
            for fr in df.fr.unique():
                for i, fl in enumerate(df.fl.unique()):
                    df_[f"p_{psi}_o_{omega}_fr_{fr}_fl_{fl}_time"] = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl"
                    ).time.to_numpy()
                    df_[f"p_{psi}_o_{omega}_fr_{fr}_fl_{fl}_overlap"] = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl"
                    ).overlap.to_numpy()
                    df_[f"p_{psi}_o_{omega}_fr_{fr}_fl_{fl}_frac_overlap"] = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl"
                    ).frac_overlap.to_numpy()
                    df_[f"p_{psi}_o_{omega}_fr_{fr}_fl_{fl}_frac_num_col_obj"] = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl"
                    ).frac_num_col_obj.to_numpy()
                    df_[f"p_{psi}_o_{omega}_fr_{fr}_fl_{fl}_volume"] = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl"
                    ).volume.to_numpy()
                    df_[f"p_{psi}_o_{omega}_fr_{fr}_fl_{fl}_num_steps"] = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl"
                    ).num_steps.to_numpy()
    df_.to_csv(os.path.join(args.input, "boxplot", f"cube_stats_r_{r}.csv"),
               index=False)
