#!/bin/bash
set -e

mkdir -p output/tmp/tikz
mkdir -p output/tikz
# cp simulation_analysis_spheres.tex output/tmp
cd output/tmp

folder="sim_120_new/0.125"

for radius in 0.5 1.0 2.0 5.0 10.0; do
   for microscope in PM LAP; do
      for species in Roden Vervet Human; do
         for model in r p; do
            for name in acc R R2; do
               cp ../../simulation_analysis_spheres.tex .
               sed -i 's=__SIMULATION__='"$folder"'=g' simulation_analysis_spheres.tex
               sed -i 's/__MICROSCOPE__/'"$microscope"'/g' simulation_analysis_spheres.tex
               sed -i 's/__SPECIES__/'"$species"'/g' simulation_analysis_spheres.tex
               sed -i 's/__MODEL__/'"$model"'/g' simulation_analysis_spheres.tex
               sed -i 's/__RADIUS__/'"$radius"'/g' simulation_analysis_spheres.tex
               sed -i 's/__NAME__/'"$name"'/g' simulation_analysis_spheres.tex

               lualatex -interaction=nonstopmode -halt-on-error --shell-escape simulation_analysis_spheres.tex
               make -j4 -f simulation_analysis_spheres.makefile
               lualatex -interaction=nonstopmode -halt-on-error --shell-escape simulation_analysis_spheres.tex
               mv simulation_analysis_spheres.pdf ../tikz/simulation_analysis_spheres_${radius}_${microscope}_${species}_${model}_${name}.pdf
               # xdg-open ../tikz/simulation_analysis_spheres_$name_$microscope_$species_$model.pdf &>/dev/null 2>&1
            done
         done
      done
   done
done
