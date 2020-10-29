# voxel size

``` sh
# simulation
python3 voxel_size.py -i ../../1_model/1_cubes/output/cube_2pop_1/cube_2pop_psi_1.00_omega_0.00_r_1.00_v0_105_.solved.h5 ../../1_model/1_cubes/output/cube_2pop_1/cube_2pop_psi_0.50_omega_90.00_r_1.00_v0_105_.solved.h5 -o output/test -p 16 -m 51 -n 10

# analyse
./voxel_size_post_0.py -i output/test -p 16
./voxel_size_post_1.py -i output/test -p 16
```
