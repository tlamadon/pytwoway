
doc:
	cp README.rst docs/source/README.rst
	rm -rf docs/build
	$(MAKE) -C docs html
	cd docs/build && \
	git init && \
	git add . && \
	git commit -m "Update documentation using Makefile" && \
	git remote add origin git@github.com:tlamadon/$(notdir $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))).git && \
	git push --force origin master:gh-pages
