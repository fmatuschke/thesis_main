for start in {0..1779..20}; do
   file=$(cat simulation_juron.sh)
   file=$(sed 's/$1/0.125/g' <<<"$file")
   # file=$(sed 's/$2/'"$radius"'/g' <<<"$file")
   file=$(sed 's/$3/'"$start"'/g' <<<"$file")

   echo "$file" >tmp.run
   echo "$file"
   #bsub < tmp.run
done
#rm tmp.run

# python3 simulation_juron.py -i \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.00*omega_0.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.30*omega_0.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.30*omega_30.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.30*omega_60.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.30*omega_90.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.50*omega_0.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.50*omega_30.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.50*omega_60.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.50*omega_90.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.60*omega_0.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.60*omega_30.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.60*omega_60.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.60*omega_90.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.90*omega_0.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.90*omega_30.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.90*omega_60.00*.solved.h5 \
# ../../1_model/1_cubes/output/cube_2pop_120/*psi_0.90*omega_90.00*.solved.h5 \
# -o output/test -v 1 --start 0
