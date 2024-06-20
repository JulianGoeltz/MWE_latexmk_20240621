tikz/fig_intuition.pdf: fig/fig_intuition.pdf
	cd ./tikz/; latexmk -pdf fig_intuition.tex

fig/fig_intuition.pdf: python/fig_intuition.py
	gridspeccer python/fig_intuition.py

change:
	sed -i 's/time/mime/g' python/fig_intuition.py

.PHONY: change
