poms.pdf: poms.tex poms.bib
	pdflatex poms.tex
	bibtex poms
	pdflatex poms.tex
	pdflatex poms.tex


clean: 
	rm poms.pdf
