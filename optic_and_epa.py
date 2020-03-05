import numpy as np
import multiprocessing
import argparse
import h5py
import os
import pandas as pd

from tqdm import tqdm

import fastpli.simulation.optic
import fastpli.analysis

parser = argparse.ArgumentParser()
parser.add_argument('--input',
                    type=str,
                    required=True,
                    nargs='+',
                    help='input h5 files')
parser.add_argument('--resolution',
                    nargs='+',
                    type=float,
                    required=True,
                    help='values of voxel sizes')
parser.add_argument('--output', type=str, required=True, help='output pkl')
parser.add_argument('--threads',
                    type=int,
                    required=False,
                    default=1,
                    help='output pkl')

args = parser.parse_args()


def _resample(data, scale):
    data_ref = np.empty(
        np.array([data.shape[0] * scale, data.shape[1] * scale, data.shape[2]],
                 int))

    if scale != 1:
        for i in range(data.shape[-1]):
            data_ref[:, :, i] = fastpli.simulation.optic.resample(
                data[:, :, i], scale)

    else:
        data_ref = data.copy()

    return data_ref


def _optic_and_epa(parameter):
    file, resolution, voxel_size_ref = parameter[0], parameter[1], parameter[2]
    df = pd.DataFrame()
    with h5py.File(file, 'r') as h5f:
        if str(min([float(i) for i in h5f])) != voxel_size_ref:
            raise ValueError("voxel_size_ref differs")

        omega = float(file.rsplit("_omega_")[-1].split("_")[0])
        f0 = float(file.rsplit("_f0_")[-1].split("_")[0])
        f1 = float(file.rsplit("_f1_")[-1].split(".simulation")[0])

        data_ref = {}
        # print(file, voxel_size_ref + '/r/simulation/data/0')
        data_ref['r'] = h5f[voxel_size_ref + '/r/simulation/data/0'][:]
        data_ref['p'] = h5f[voxel_size_ref + '/p/simulation/data/0'][:]

        for voxel_size in h5f:
            for model in h5f[voxel_size]:
                dset = h5f[voxel_size + '/' + model]
                data = dset['simulation/data/0'][:]

                for res in resolution:
                    # rescale
                    scale = float(voxel_size_ref) / float(res)
                    data_ref_optic_r = _resample(data_ref['r'], scale)
                    ref_r_trans, ref_r_dirc, ref_r_ret = fastpli.analysis.epa.epa(
                        data_ref_optic_r)

                    scale = float(voxel_size_ref) / float(res)
                    data_ref_optic_p = _resample(data_ref['p'], scale)
                    ref_p_trans, ref_p_dirc, ref_p_ret = fastpli.analysis.epa.epa(
                        data_ref_optic_p)

                    scale = float(voxel_size) / float(res)
                    data_optic = _resample(data, scale)
                    trans, dirc, ret = fastpli.analysis.epa.epa(data_optic)

                    if not np.array_equal(
                            data_optic.shape,
                            data_ref_optic_r.shape) or not np.array_equal(
                                data_optic.shape, data_ref_optic_p.shape):
                        raise ValueError(
                            f"Wrong scale: {data_optic.shape}, {data_ref_optic_r.shape}, {data_ref_optic_p.shape}"
                        )

                    df = df.append(
                        {
                            'voxel_size':
                                float(voxel_size),
                            'model':
                                model,
                            'resolution':
                                res,
                            'omega':
                                float(omega),
                            'f0':
                                f0,
                            'f1':
                                f1,
                            # 'data':
                            #     data.flatten().tolist(),
                            'transmittance':
                                trans.flatten().tolist(),
                            'direction':
                                dirc.flatten().tolist(),
                            'retardation':
                                ret.flatten().tolist(),
                            'transmittance_ref_r':
                                ref_r_trans.flatten().tolist(),
                            'direction_ref_r':
                                ref_r_dirc.flatten().tolist(),
                            'retardation_ref_r':
                                ref_r_ret.flatten().tolist(),
                            'transmittance_ref_p':
                                ref_p_trans.flatten().tolist(),
                            'direction_ref_p':
                                ref_p_dirc.flatten().tolist(),
                            'retardation_ref_p':
                                ref_p_ret.flatten().tolist()
                        },
                        ignore_index=True)
    return df


def optic_and_epa(input, output, resolution):
    if not ".pkl" in output and not ".csv" in output:
        raise ValueError("output not recognized: " + output)

    resolution = np.sort(resolution)

    df = pd.DataFrame()
    voxel_size_ref = str(min([float(i) for i in h5py.File(input[0], 'r')]))

    with multiprocessing.Pool(processes=args.threads) as pool:
        parameter = [(file, resolution, voxel_size_ref) for file in input]
        dfs = [
            d for d in tqdm(pool.imap(_optic_and_epa, parameter),
                            total=len(parameter))
        ]
        for d in dfs:
            df = df.append(d)

    if ".pkl" in output:
        df.to_pickle(output)
    elif ".csv" in output:
        df.to_csv(output)
    else:
        raise ValueError("output not recognized")


if __name__ == "__main__":

    files = args.input
    output = args.output

    # files.sort()
    # file_name = files[0]
    # file_name = file_name.rpartition(".simulation")[0] + '.analysis.pkl'
    optic_and_epa(input=args.input,
                  output=args.output,
                  resolution=args.resolution)
