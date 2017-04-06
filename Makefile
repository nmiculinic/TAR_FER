paper: paper/main.pdf

paper/main.pdf: paper/main.tex
	cd paper && latexmk -pdf main.tex

data: FORCE
	python3 data/make_data.py

FORCE:



