PAPER = Morphogenic_Intelligence
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex sections/*.tex references.bib
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER) || true
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

view: $(PAPER).pdf
	open $(PAPER).pdf

clean:
	rm -f $(PAPER).pdf $(PAPER).aux $(PAPER).log $(PAPER).bbl $(PAPER).blg $(PAPER).out $(PAPER).toc $(PAPER).fls $(PAPER).fdb_latexmk
	rm -f sections/*.aux

distclean: clean
	rm -f *~ sections/*~
