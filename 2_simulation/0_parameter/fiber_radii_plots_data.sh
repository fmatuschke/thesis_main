#!/bin/bash
set -e

# rm -rf output/tmp
mkdir -p output/tmp
mkdir -p output/tikz

# lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp -shell-escape fiber_radii_lap_plots_data_0.tex
# lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp -shell-escape fiber_radii_lap_plots_data_0.tex
# mv output/tmp/fiber_radii_lap_plots_data_0.pdf output/tikz/fiber_radii_lap_plots_data_0.pdf
# xdg-open output/tikz/fiber_radii_lap_plots_data_0.pdf &>/dev/null 2>&1

# lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp -shell-escape fiber_radii_lap_plots_data_1.tex
# lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp -shell-escape fiber_radii_lap_plots_data_1.tex
# mv output/tmp/fiber_radii_lap_plots_data_1.pdf output/tikz/fiber_radii_lap_plots_data_1.pdf
# xdg-open output/tikz/fiber_radii_lap_plots_data_1.pdf &>/dev/null 2>&1

lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp -shell-escape fiber_radii_plots_data_0.tex
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp -shell-escape fiber_radii_plots_data_0.tex
mv output/tmp/fiber_radii_plots_data_0.pdf output/tikz/fiber_radii_plots_data_0.pdf
xdg-open output/tikz/fiber_radii_plots_data_0.pdf &>/dev/null 2>&1

lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp -shell-escape fiber_radii_plots_data_1.tex
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp -shell-escape fiber_radii_plots_data_1.tex
mv output/tmp/fiber_radii_plots_data_1.pdf output/tikz/fiber_radii_plots_data_1.pdf
xdg-open output/tikz/fiber_radii_plots_data_1.pdf &>/dev/null 2>&1