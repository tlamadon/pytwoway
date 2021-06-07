#GIT_REMOTE =  https://github.com/tlamadon/pytwoway.git
GIT_ORIGIN=`git remote -v | head -1`

doc:
	cp README.rst docs/source/README.rst
	rm -rf docs/build
	$(MAKE) -C docs html
	cd docs/build/html && \
	git init && \
	git add . && \
	git commit -m "Update documentation using Makefile" && \
	git remote add $GIT_ORIGIN && \
	git push --force origin master:gh-pages
	rm -rf docs/build
