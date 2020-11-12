for v in 1.00 0.50 2.00 5.00 10.00
do
   for start in 0 20 40 60 80 100 120 140 160 180
   do
      file=$(cat simulation_juron.sh)
      file=$(sed 's/$1/0.125/g' <<<"$file")
      file=$(sed 's/$2/'"$v"'/g' <<<"$file")


      file=$(sed 's/$3/'"$start"'/g' <<<"$file")

      echo "$file" >tmp.run
      echo "$file"
      bsub < tmp.run
   done
done
rm tmp.run
