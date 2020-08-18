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

# # missing members
# for r in df.r.unique():
#     for omega in df.omega.unique():
#         for psi in df[df.omega == omega].psi.unique():
#             for fr in df.fr.unique():
#                 for fl in df.fl.unique():
#                     for n in df.n.unique():
#                         if not len(
#                                 df.query(
#                                     "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl & n == @n"
#                                 )):
#                             raise ValueError("FOO")
#                             # time = 60 * 60 * 11
#                             # frac_num_col_obj = 0.5
#                             # volume = 0.5
#                             # ["time", "frac_num_col_obj", "volume"]
#                             # df = df.append(
#                             #     {
#                             #         "r": r,
#                             #         "psi": psi,
#                             #         "omega": omega,
#                             #         "fr": fr,
#                             #         "fl": fl,
#                             #         "time": time,
#                             #         "n": n,
#                             #         "frac_num_col_obj": frac_num_col_obj,
#                             #         "volume": volume
#                             #     },
#                             #     ignore_index=True)

#                     # ys = [
#                     #     "time", "overlap", "frac_overlap", "frac_num_col_obj",
#                     #     "volume", "num_steps"
#                     # ]
#                     # # ys = ["time", "frac_num_col_obj", "volume"]
#                     # df_ = df.query(
#                     #     "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl"
#                     # )
#                     # df_[ys].to_csv(
#                     #     os.path.join(
#                     #         args.input,
#                     #         f'cube_stat_r_{r}_psi_{psi}_fr_{fr}_fl_{fl}_.csv'),
#                     #     index=False,
#                     #     #   float_format='%.6f'
#                     # )

# df = df[["n", "r", "omega", "psi", "fr", "fl", "time"]]

# # print(list(df.columns).remove("n"))

# print(df)

# df.pivot(index="n", columns=["r", "omega", "psi", "fr", "fl"],
#          values="time").to_csv(f"test.csv")

#  .to_csv(f"test.csv")

# print(df.fr.unique())
# print(df.fl.unique())
# print(df.psi.unique())

for r in df.r.unique():
    df_ = pd.DataFrame()
    for omega in df.omega.unique():
        for psi in df[df.omega == omega].psi.unique():
            for fr in df.fr.unique():
                for i, fl in enumerate(df.fl.unique()):
                    df_[f"p_{psi}_fr_{fr}_fl_{fl}_time"] = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl"
                    ).time.to_numpy()
                    df_[f"p_{psi}_fr_{fr}_fl_{fl}_overlap"] = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl"
                    ).overlap.to_numpy()
                    df_[f"p_{psi}_fr_{fr}_fl_{fl}_frac_overlap"] = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl"
                    ).frac_overlap.to_numpy()
                    df_[f"p_{psi}_fr_{fr}_fl_{fl}_frac_num_col_obj"] = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl"
                    ).frac_num_col_obj.to_numpy()
                    df_[f"p_{psi}_fr_{fr}_fl_{fl}_volume"] = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl"
                    ).volume.to_numpy()
                    df_[f"p_{psi}_fr_{fr}_fl_{fl}_num_steps"] = df.query(
                        "r == @r & omega == @omega & psi == @psi & fr == @fr & fl == @fl"
                    ).num_steps.to_numpy()
    df_.to_csv(f"output/tmp/cube_stats_r_{r}.csv", index=False)
