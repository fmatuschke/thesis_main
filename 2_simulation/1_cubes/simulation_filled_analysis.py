import os

import numpy as np
import pandas as pd
import tqdm

import helper.spherical_harmonics
import helper.schilling
import polar_hist_to_tikz

path = 'output/sim_120_ime_filled'
df = pd.read_pickle(os.path.join(path, "analysis", f"cube_2pop_simulation.pkl"))
df["trel_mean"] = df["rofl_trel"].apply(lambda x: np.mean(x))
df["ret_mean"] = df["epa_ret"].apply(lambda x: np.mean(x))
df["trans_mean"] = df["epa_trans"].apply(lambda x: np.mean(x))

microscope = 'PM'
species = 'Vervet'
df = df.query('microscope == @microscope & species == @species')
df_ = []
for index, row in tqdm.tqdm(df[df.model == 'r'].iterrows(),
                            total=len(df[df.model == 'r'])):
    radius, omega, psi, f0_inc, f1_rot = row['radius'], row['omega'], row[
        'psi'], row['f0_inc'], row['f1_rot']

    phi = row['rofl_dir'].ravel()
    theta = np.pi / 2 - row['rofl_inc'].ravel()
    sh_filled = helper.spherical_harmonics.real_spherical_harmonics(
        phi, theta, 6)

    phi = df.iloc[index + 1]['rofl_dir'].ravel()
    theta = np.pi / 2 - df.iloc[index + 1]['rofl_inc'].ravel()
    sh = helper.spherical_harmonics.real_spherical_harmonics(phi, theta, 6)

    df_.append(
        pd.DataFrame(
            {
                "radius":
                    radius,
                "omega":
                    omega,
                "psi":
                    psi,
                "f0_inc":
                    f0_inc,
                "f1_rot":
                    f1_rot,
                "acc":
                    helper.schilling.angular_correlation_coefficient(
                        sh, sh_filled)
            },
            index=[0]))
df_ = pd.concat(df_, ignore_index=True)

for radius in df_.radius.unique():
    polar_hist_to_tikz.generate(df_[df_.radius == radius],
                                "acc",
                                f"simulation_analysis_filled_hist_{radius}_acc",
                                crange=[0.95, 1],
                                psi_list=[0.30, 0.50, 0.60, 0.90],
                                f0_list=[0, 30, 60, 90])
