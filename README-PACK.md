# pytwoway packaging procedures

pytwoway is a two way fixed effect models in python

# Package pytwoway for PIP

PIP directions from the [Packaging Python page](https://packaging.python.org/tutorials/packaging-projects/)

- Packaging a test wheel:
  - from the repository root directory
  - create your custom setup file, using your own user name:
	```bash
	sed 's/__user__/ENTER_YOUR_NAME_HERE/g' setup-test.py.default > setup-test.py
	```

  - build the wheel:
	```bash
	python3 setup-test.py sdist bdist_wheel
	```

  - register an account on [Test Py Pi](https://test.pypi.org/account/register/ )
  - create an API token
  - copy the token for later
  - upload the distribution onto Test Py Pi, using Twine:
	```bash
	python3 -m pip install --user --upgrade twine
	python3 -m twine upload --repository testpypi dist/*
	```
    use `__token__` for username and your token for password (including the `pypi-` prefix)
  - test and install using:
	```bash
	python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps pytwoway-test-YOUR-USERNAME-HERE
	```
	alternatively, you can try installing the dependencies
	```bash
	python3 -m pip install --index-url https://test.pypi.org/simple/ pytwoway-test-YOUR-USERNAME-HERE
	```
	And complement with the real pip
	```bash
	pip install MISSING_DEP
	```
  - run a test:
	```bash
	python3 tests/test_simu.py 
	```

- Packaging the final distribution:
```bash
to come ...
```

# Package pytwoway for CONDA
