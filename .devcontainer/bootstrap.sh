#!/bin/bash

# `bootstrap.sh` is run after the host directory is mounted to the container.
# It is a good place to perform actions that require this repository to be mounted.
# Ex. Installing a local python module with `pip install .`.

echo "Bootstrapping the container!"

echo "Installing CyberBattleSim!"
cd CyberBattleSim
bash -i install-conda.sh
bash -i init.sh

# Allow python to always find CyberBattleSim modules.
echo "../../../../../../reinforced-butter/CyberBattleSim" > ~/miniconda3/envs/cybersim/lib/python3.10/site-packages/CyberBattleSim.pth

echo "conda activate cybersim" >> ~/.bashrc
