#!/bin/bash

# `bootstrap.sh` is run after the host directory is mounted to the container.
# It is a good place to perform actions that require this repository to be mounted.
# Ex. Installing a local python module with `pip install .`.

echo "Bootstrapping the container!"
