import h5py

import os

import glob
import tqdm

for file in tqdm.tqdm(glob.glob('output/cube_2pop_135_rc1.a/*.h5')):

    with h5py.File(file, 'r+') as h5f:

        # try:
        #     del h5f['PM']
        # except:
        #     pass
        # try:
        #     del h5f['PM/Vervet']
        # except:
        #     pass
        # try:
        #     del h5f['PM/Human']
        # except:
        #     pass
        # try:
        #     del h5f['PM/Roden']
        # except:
        #     pass
        # try:
        #     del h5f['pm/Vervet']
        # except:
        #     pass
        # try:
        #     del h5f['pm/Human']
        # except:
        #     pass
        # try:
        #     del h5f['pm/Roden']
        # except:
        #     pass

        h5f['/PM/Vervet'] = h5py.SoftLink('/pm/vervet')
        h5f['/PM/Human'] = h5py.SoftLink('/pm/human')
        h5f['/PM/Roden'] = h5py.SoftLink('/pm/roden')
