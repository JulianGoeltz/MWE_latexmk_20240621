tikz/fig_intuition/fig_intuition.pdf: fig/fig.pdf
	cd ./tikz/fig_intuition; latexmk -pdf fig_intuition.tex

fig/fig.pdf: python/fig.py
	cd python; gridspeccer --mplrc matplotlibrc --loglevel WARNING fig.py

change:
	sed -i 's/time/mime/g' python/fig.py

clean:
	$(RM) fig/*.pdf
	cd ./tikz/fig_intuition; latexmk -pdf fig_intuition.tex -C

.PHONY: change
