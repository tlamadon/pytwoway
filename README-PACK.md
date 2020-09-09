# pytwoway packaging procedures

pytwoway is a two way fixed effect models in python

## Package pytwoway for PIP

PIP directions from the [Packaging Python page](https://packaging.python.org/tutorials/packaging-projects/)

### Packaging a test wheel
- from the repository root directory
- create your custom setup file, using your own user name:
  ```bash
  sed 's/__user__/YOUR_USERNAME_HERE/g' setup-test.py.default > setup-test.py
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
- test installation using:
  ```bash
  python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps pytwoway-test-YOUR_USERNAME_HERE
  ```
  alternatively, you can try installing the dependencies
  ```bash
  python3 -m pip install --index-url https://test.pypi.org/simple/ pytwoway-test-YOUR_USERNAME_HERE
  ```
  and complement with the real pip
  ```bash
  pip install MISSING_DEP
  ```
- run a test:
  ```bash
  python3 tests/test_simu.py 
  ```

### Packaging the final distribution
- from the repository root directory
- build the wheel:
  ```bash
  python3 setup.py sdist bdist_wheel
  ```

- register an account on [Py Pi](https://pypi.org/account/register/ )
- create an API token
- copy the token for later
- upload the distribution onto Py Pi, using Twine:
  ```bash
  python3 -m pip install --user --upgrade twine
  python3 -m twine upload dist/*
  ```
  use `__token__` for username and your token for password (including the `pypi-` prefix)
  
### Installing distribution

  ```bash
  pip install pytwoway
  ```

### Running a test

  ```bash
  python3 tests/test_simu.py 
  ```

## Package pytwoway for CONDA

Directions from [Building conda packages](https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/building-conda-packages.html)

You can use the `meta.yaml` file provided or generate it using skeleton.

### Prepare the recipe using skeleton (optional)

We will simply use the PIP package and convert it using skeleton.

- install conda-build:
  ```bash
  conda install conda-build
	
- go to working directory to avoid name conflicts
  ```bash
  cd recipe4conda
  ```
	
- prepare the recipe:
  ```bash
  conda skeleton pypi pytwoway
  cd ..
  ```
	
### Building using conda-build:
	
- build the conda package based on the provided yaml or the one written with skeleton.
  we need to look for *pyreadr* module through conda-forge or a *conda.exceptions.ResolvePackageNotFound* will be thrown:
  ```bash
  conda build -c conda-forge recipe4conda 
  ```
  
  or using the recipe made with skeleton:
  
  ```bash
  conda build -c conda-forge recipe4conda/pytwoway 
  ```