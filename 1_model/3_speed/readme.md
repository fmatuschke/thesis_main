```
export OMP_PLACES="cores"
python3 speedup.py -i \
   ../1_cubes/output/cube_2pop_135_rc1/cube_2pop_psi_0.00_omega_0.00_r_0.50_v0_135_.solved.h5 \
   ../1_cubes/output/cube_2pop_135_rc1/cube_2pop_psi_0.50_omega_90.00_r_0.50_v0_135_.solved.h5 \
   -r 100 -a 0 100 1000 10000 -o output/r_0.5_.pkl

python3 speedup.ipy
./plot.sh
```
