poetry env use 3.9 && \
    poetry install && \
    poetry run python -m ipykernel install --user --name pytwoway
