import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()
    
print("setup.py: Cleaning distribution and build directories")
os.system("rm -R dist; rm -R build; rm -R *.egg-info")

print(setuptools.find_packages())  
packages = ['pytwoway'] # setuptools.find_packages()

setuptools.setup(
    name="pytwoway",
    version="0.0.1",
    maintainer="Thibaut Lamadon",
    author="Thibaut Lamadon, Adam Alexander Oppenheimer",
#    author_email="thibaut.lamadon@gmail.com",
    license="Apache Software License ",
    keywords="sparse,two way fixed effects",
    description="Two way fixed effect models in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tlamadon/pytwoway",
    packages=packages,
	classifiers=[
		"Development Status :: 2 - Pre-Alpha ",
		"Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
	    "Topic :: Scientific/Engineering :: Mathematics",
		"Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
		"Intended Audience :: Developers",
	    "Intended Audience :: Science/Research"
    ],
    project_urls={
        "Documentation": "https://github.com/tlamadon/pytwoway",
        "Source": "https://github.com/tlamadon/pytwoway",
        "Tracker": "https://github.com/tlamadon/pytwoway/issues",
    },    
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'networkx>=2.3',
        'scikit-learn',
        'scipy',
        'pyamg',
        'pyreadr',
        'tqdm',
        'decorator'
   ]
)
