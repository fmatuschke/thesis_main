#!/bin/bash
set -e

mkdir -p output/tmp/tikz
mkdir -p output/tikz
# cp simulation_analysis_spheres.tex output/tmp
cd output/tmp

radius=0.5
microscope=PM
species=Vervet
model=r
name=acc
norm=False

cp ../../simulation_analysis_hist.tex .
rm -rf tikz/*
sed -i 's?__PATH__?'"$1"'?g' simulation_analysis_hist.tex
sed -i 's?__MICROSCOPE__?'"$microscope"'?g' simulation_analysis_hist.tex
sed -i 's?__SPECIES__?'"$species"'?g' simulation_analysis_hist.tex
sed -i 's?__MODEL__?'"$model"'?g' simulation_analysis_hist.tex
sed -i 's?__RADIUS__?'"$radius"'?g' simulation_analysis_hist.tex
sed -i 's?__NAME__?'"$name"'?g' simulation_analysis_hist.tex
sed -i 's?__NORM__?'"$norm"'?g' simulation_analysis_hist.tex

lualatex -interaction=nonstopmode -halt-on-error --shell-escape simulation_analysis_hist.tex
# make -j4 -f simulation_analysis_hist.makefile
# lualatex -interaction=nonstopmode -halt-on-error --shell-escape simulation_analysis_hist.tex
mv simulation_analysis_hist.pdf ../tikz/simulation_analysis_hist_${radius}_${microscope}_${species}_${model}_${name}_${norm}.pdf

xdg-open ../tikz/simulation_analysis_hist_${radius}_${microscope}_${species}_${model}_${name}_${norm}.pdf
# exit 1

# for radius in 0.5 1.0 2.0 5.0 10.0; do
#    for microscope in PM LAP; do
#       for species in Roden Vervet Human; do
#          for model in r p; do
#             for name in acc; do
#                for norm in False; do

for radius in 0.5; do
   for microscope in PM; do
      for species in Vervet; do
         for model in r; do
            for name in acc; do
               for norm in False; do

                  cp ../../simulation_analysis_hist.tex .
                  rm -rf tikz/*
                  sed -i 's?__PATH__?'"$1"'?g' simulation_analysis_hist.tex
                  sed -i 's?__MICROSCOPE__?'"$microscope"'?g' simulation_analysis_hist.tex
                  sed -i 's?__SPECIES__?'"$species"'?g' simulation_analysis_hist.tex
                  sed -i 's?__MODEL__?'"$model"'?g' simulation_analysis_hist.tex
                  sed -i 's?__RADIUS__?'"$radius"'?g' simulation_analysis_hist.tex
                  sed -i 's?__NAME__?'"$name"'?g' simulation_analysis_hist.tex
                  sed -i 's?__NORM__?'"$norm"'?g' simulation_analysis_hist.tex

                  lualatex -interaction=nonstopmode -halt-on-error --shell-escape simulation_analysis_hist.tex
                  # make -j4 -f simulation_analysis_hist.makefile
                  # lualatex -interaction=nonstopmode -halt-on-error --shell-escape simulation_analysis_hist.tex
                  mv simulation_analysis_hist.pdf ../tikz/simulation_analysis_hist_${radius}_${microscope}_${species}_${model}_${name}_${norm}.pdf

                  # xdg-open ../tikz/simulation_analysis_hist_${radius}_${microscope}_${species}_${model}_${name}_${norm}.pdf
                  # exit 1
               done
            done
         done
      done
   done
done
