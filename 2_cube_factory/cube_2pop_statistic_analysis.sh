
mkdir -p output/tikz/tmp
mkdir -p output/tikz/output/tikz/tmp
pdflatex -interaction=nonstopmode -halt-on-error --shell-escape -output-directory=output/tikz cube_2pop_statistic_analysis.tex > /dev/null
# pdftk cube_2pop_statistic_analysis.pdf burst output cube_2pop_statistic_analysis_%01d.pdf
