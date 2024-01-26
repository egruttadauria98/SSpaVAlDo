# To build:
# From folder containing this Dockerfile
# docker build --tag study_001:0.1 --build-arg USER=<USER_NAME> --build-arg UID=$(id -u) .

# To run Jupyter notebook server:
# docker run --gpus all -p 8888:8888 --hostname study_001 --name study_001 -v /home/essid:/home/essid -v /home/essid/data:/home/essid/data study_001:0.1 start-notebook.sh

# To properly run bash (with sudo granted to essid user):
# docker run --gpus all --rm -ti -u root --hostname study_001 --name study_001 -v /home/essid:/home/essid -v /home/essid/data:/home/essid/data study_001:0.1 start.sh

# For ssh use user root and start-ssh.sh
# docker run --gpus all --group-add users --user root:16229 --rm -ti --hostname dns --name dns -v /tsi/audiosig/audible:/home/essid/data -v /cal/homes/essid:/home/essid/code study_001:0.1 start-ssh.sh

FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ARG USER
ARG UID="1000"

ENV HOME /home/${USER}
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

ENV MINICONDA_VERSION=py310_22.11.1-1 \
    CONDA_VERSION=23.1.0 \
    CONDA_DIR=/opt/conda \
    CONDA_AUDIBLE_ENV=audible

ENV AUDIBLE_CODE=$HOME/audible
ENV AUDIBLE_DATA=$HOME/data

# User for jupyter notebook setup: $NB_USER (var used in start.sh, see below)
ENV NB_USER=${USER}
ENV NB_UID=${UID}
ENV NB_GID="100"
ENV GRANT_SUDO=yes

USER root

RUN apt-get update --fix-missing && apt-get install -y wget ca-certificates \
    sudo iproute2 net-tools build-essential inetutils-ping\
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    bzip2 git mercurial subversion \
    zile htop screen tmux openssh-server sshfs

# ffmpeg for matplotlib anim & dvipng for latex labels
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y ffmpeg dvipng && \
    rm -rf /var/lib/apt/lists/*

# -- System -------------------------------------------------------------------
# Much of the following is from https://github.com/jupyter/docker-stacks/blob/master/base-notebook/Dockerfile

# Copy a script that we will use to correct permissions after running certain commands
COPY ./fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc

# Create NB_USER with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su && \
    sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers && \
    sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers && \
    useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown $NB_USER:$NB_GID $CONDA_DIR && \
    chmod g+w /etc/passwd && \
    fix-permissions $HOME && \
    fix-permissions $CONDA_DIR

# Miniconda -------------------------------------------------------------------
USER $NB_UID
WORKDIR $HOME

# Install conda as NB_USER and check the md5 sum provided on the download site
RUN cd /tmp && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    /bin/bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh

USER root
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

USER $NB_UID
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate $CONDA_AUDIBLE_ENV" >> ~/.bashrc && \
    conda clean --all -f -y && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install Tini
RUN conda install --quiet --yes 'tini' && \
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# -- Install conda environment-------------------------------------------------
USER $NB_UID

COPY environment.yml .
RUN conda env create && \
    conda clean --all -f -y && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# -- Additional packages using pip --------------------------------------------
# COPY requirements.txt /tmp/
# RUN pip install -r /tmp/requirements.txt

# -- Install Jupyter Notebook, Lab, and Hub -----------------------------------
# Generate a notebook server config
# Cleanup temporary files
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "audible", "/bin/bash", "-c"]

RUN conda install --quiet --yes \
    'matplotlib' \
    'notebook' \
    'jupyterhub' \
    'widgetsnbextension' \
    'jupyterlab' && \
    conda clean --all -f -y && \
    npm cache clean --force && \
    jupyter notebook --generate-config && \
    rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Activate ipywidgets extension in the environment that runs the notebook server
RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
    # Also activate ipywidgets extension for JupyterLab
    # Check this URL for most recent compatibilities
    # https://github.com/jupyter-widgets/ipywidgets/tree/master/packages/jupyterlab-manager
    # jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build && \
    # # jupyter labextension install @bokeh/jupyter_bokeh --no-build && \
    # jupyter labextension install jupyter-matplotlib --no-build && \
    # jupyter lab build -y --dev-build=False --minimize=False && \
    jupyter lab clean -y && \
    npm cache clean --force && \
    rm -rf $HOME/.cache/yarn && \
    rm -rf $HOME/.node-gyp && \
    fix-permissions $CONDA_DIR && \
    fix-permissions $HOME

# Install facets which does not have a pip or conda package at the moment
# RUN cd /tmp && \
#     git clone https://github.com/PAIR-code/facets.git && \
#     cd facets && \
#     jupyter nbextension install facets-dist/ --sys-prefix && \
#     cd && \
#     rm -rf /tmp/facets && \
#     fix-permissions $CONDA_DIR && \
#     fix-permissions $HOME

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME $HOME/.cache/
RUN MPLBACKEND=Agg $CONDA_DIR/envs/audible/bin/python -c "import matplotlib.pyplot" && \
    fix-permissions $HOME

# Choose to use jupyter lab
ENV JUPYTER_ENABLE_LAB=1


# -- Startup env --------------------------------------------------------------
ENTRYPOINT ["tini", "-g", "--"]

EXPOSE 8888
EXPOSE 22

USER root

# Copy local files as late as possible to avoid cache busting
COPY start.sh start-ssh.sh start-notebook.sh start-singleuser.sh /usr/local/bin/
RUN chmod a+rx /usr/local/bin/start.sh
RUN chmod a+rx /usr/local/bin/start-notebook.sh
RUN chmod a+rx /usr/local/bin/start-singleuser.sh

COPY jupyter_notebook_config.py /etc/jupyter/
# Fix permissions on /etc/jupyter as root
RUN fix-permissions /etc/jupyter/

# CMD ["start-notebook.sh"]

# Switch back to essid to avoid accidental container runs as root
USER $NB_UID
