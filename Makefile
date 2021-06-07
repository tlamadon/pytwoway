
doc:
	rm -rf docs/build
	cp README.rst docs/source/README.rst
	$(MAKE) -C docs html copy
	dirname = ${PWD##*/}
	dirname = "${dirname%"${dirname##*[!/]}"}"
	dirname = "${dirname##*/}"
	cd docs/build && \
	git init && \
	git add . && \
	git commit -m "Update documentation using Makefile" && \
	git remote add origin git@github.com:tlamadon/${dirname}.git && \
	git push --force origin master:gh-pages
