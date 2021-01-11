#!/bin/bash
set -e

mkdir -p output/tmp/tikz
mkdir -p output/tikz
# cp simulation_analysis_spheres.tex output/tmp
cd output/tmp

for radius in 0.5 1.0 2.0 5.0 10.0; do
   for microscope in PM LAP; do
      for species in Roden Vervet Human; do
         for model in r p; do
            for name in acc R R2; do
               cp ../../simulation_analysis_spheres_.tex .
               rm -rf tikz/*
               sed -i 's/__MICROSCOPE__/'"$microscope"'/g' simulation_analysis_spheres_.tex
               sed -i 's/__SPECIES__/'"$species"'/g' simulation_analysis_spheres_.tex
               sed -i 's/__MODEL__/'"$model"'/g' simulation_analysis_spheres_.tex
               sed -i 's/__RADIUS__/'"$radius"'/g' simulation_analysis_spheres_.tex
               sed -i 's/__NAME__/'"$name"'/g' simulation_analysis_spheres_.tex

               lualatex -interaction=nonstopmode -halt-on-error --shell-escape simulation_analysis_spheres_.tex
               make -j4 -f simulation_analysis_spheres_.makefile
               lualatex -interaction=nonstopmode -halt-on-error --shell-escape simulation_analysis_spheres_.tex
               mv simulation_analysis_spheres_.pdf ../tikz/simulation_analysis_hist_${radius}_${microscope}_${species}_${model}_${name}.pdf
            done
         done
      done
   done
done
