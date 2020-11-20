for start in {0..2719..20}
do
   file=$(cat simulation_juron.sh)
   file=$(sed 's/$1/0.125/g' <<<"$file")
   file=$(sed 's/$2/'"$start"'/g' <<<"$file")

   echo "$file" > simulation_juron.run
   echo "$file"
   # bsub < simulation_juron.run
done
