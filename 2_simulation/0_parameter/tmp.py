import os
import glob
import numpy as np
import h5py
import tqdm
import tifffile as tiff

files = glob.glob(
    "/data/PLI-LAP2/TESTREIHEN/WiederholungsmessungGrauabstufung_LAP_20201106/*.tif"
)

with h5py.File("/data/PLI-Group/felix/data/thesis/data/LAP/test500.h5",
               "w") as h5f:
    im = tiff.imread(files[0])
    print(im.dtype)
    print(im.shape)
    data = np.empty((im.shape[0], im.shape[1], 500), dtype=im.dtype)
    # dset = h5f.create_dataset("data", (im.shape[0], im.shape[1], 500),
    #                           dtype=im.dtype)

    for i, file in tqdm.tqdm(enumerate(files), total=len(files)):
        data[:, :, i] = tiff.imread(file)[:, :, 1]
    h5f['data'] = data
