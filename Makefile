paper: paper/main.pdf

paper/main.pdf: paper/main.tex
	cd paper && latexmk -pdf main.tex
