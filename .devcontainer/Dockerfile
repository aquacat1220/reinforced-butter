FROM python:3.10

RUN apt-get update && apt-get upgrade -y

# Remove existing non-root user.
RUN if (id -nu 1000); then userdel -r $(id -nu 1000); fi

# Setup non-root user.
ARG host_username=""
ENV HOST_USERNAME=${host_username:-anonymous_kitten}

RUN useradd --create-home --shell /bin/bash -G sudo ${HOST_USERNAME}
RUN apt-get install -y sudo
RUN echo "${HOST_USERNAME}:${HOST_USERNAME}" | chpasswd

# ---------------- Start commands to run as root ----------------

# Following "https://cage-challenge.github.io/cage-challenge-4/pages/tutorials/01_Getting_Started/1_Introduction/".
RUN apt install python3-tk --assume-yes

# ---------------- End commands to run as root ----------------

USER ${HOST_USERNAME}
WORKDIR /home/${HOST_USERNAME}

# ---------------- Start commands to run as user ----------------

RUN mkdir -p ~/.local/bin

# ---------------- End commands to run as user ----------------

# Entrypoint is an interactive shell.
ENTRYPOINT ["/bin/bash", "-i"]