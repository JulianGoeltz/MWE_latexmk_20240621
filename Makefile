tikz/fig_intuition/fig_intuition.pdf: fig/fig_intuition.pdf
	cd ./tikz/fig_intuition; latexmk -pdf fig_intuition.tex

fig/fig_intuition.pdf: python/fig_intuition.py
	cd python; gridspeccer --mplrc matplotlibrc --loglevel WARNING fig_intuition.py

change:
	sed -i 's/time/mime/g' python/fig_intuition.py

clean:
	$(RM) fig/*.pdf
	cd ./tikz/fig_intuition; latexmk -pdf fig_intuition.tex -C

.PHONY: change
