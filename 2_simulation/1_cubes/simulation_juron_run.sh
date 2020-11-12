for v in 1.00 0.50 2.00 5.00 10.00
do
   file=$(cat simulation_juron.sh)
   file=$(sed 's/$1/0.125/g' <<<"$file")
   file=$(sed 's/$2'"$v"'g' <<<"$file")

   echo "$file" >tmp.run
   echo "$file"
   bsub < tmp.run
done
rm tmp.run
