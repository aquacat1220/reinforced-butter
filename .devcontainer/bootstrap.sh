#!/bin/bash

# `bootstrap.sh` is run after the host directory is mounted to the container.
# It is a good place to perform actions that require this repository to be mounted.
# Ex. Installing a local python module with `pip install .`.

echo "Bootstrapping the container!"

echo "Installing CybOrg for the first time, following instructions from \"https://cage-challenge.github.io/cage-challenge-4/pages/tutorials/01_Getting_Started/1_Introduction/\""

cd external/cage-challenge-4
pip install -r Requirements.txt
pip install -e .