
# MADT: Offline Pre-trained Multi-Agent Decision Transformer

A link to our paper can be found on [Arxiv](https://arxiv.org/abs/2112.02845).

## Overview

Official codebase for Offline Pre-trained Multi-Agent Decision Transformer.
Contains scripts to reproduce experiments.

![image info](./architecture.png)

## Instructions

We provide code in two sub-directories: `atari` containing code for Atari experiments and `gym` containing code for OpenAI Gym experiments.
See corresponding READMEs in each folder for instructions; scripts should be run from the respective directories.
It may be necessary to add the respective directories to your PYTHONPATH.

The offline smac dataset for this repo is available at [here](https://linghui.cowtransfer.com/s/54a722196db143).
```shell
## password is z0j559
```

## How to run experiments
1. setup python environment with 'requirements.txt'
2. to install StarCraft II & SMAC, you could run 'bash install_sc2.sh'. Or you could install them manually to other path you like, following the official link: https://github.com/oxwhirl/smac.
2. enter the 'sc2' folder.
3. set hyper-parameters in 'run_madt_sc2.py' line 19-52 according to appendix.
4. select a maps to test in 'envs/config.py' line 142
5. run the 'run_madt_sc2.py' script

