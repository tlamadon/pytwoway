FROM docker.io/jupyter/datascience-notebook:latest

#FROM docker.io/jupyter/julia-notebook

USER root
RUN apt-get update && \
    apt-get install -y dos2unix software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.9 python3.9-distutils && \
    pip install poetry

# # Generally, Dev Container Features assume that the non-root user (in this case jovyan)
# # is in a group with the same name (in this case jovyan). So we must first make that so.
RUN groupadd jovyan \
    && usermod -g jovyan -a -G users jovyan

USER jovyan

# # tells julia to use conda (since it is already there)
# ENV JULIA_CONDAPKG_EXE=conda

# tells Jupyterlab not to ask for a token/password, you can change that is you want.
CMD ["start-notebook.sh" ,"--IdentityProvider.token=''"]