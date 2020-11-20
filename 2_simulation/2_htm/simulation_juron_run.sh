for start in {0..680..20}
do
   file=$(cat simulation_juron.sh)
   file=$(sed 's/$1/0.125/g' <<<"$file")
   file=$(sed 's/$2/'"$start"'/g' <<<"$file")

   echo "$file" >tmp.run
   echo "$file"
   bsub < tmp.run
done
rm tmp.run
