#!/bin/bash

enable_modules
module load StdEnv/2020 gcc/8.4.0
module load python/3.9.6

if [[ ! -d "python_venv" ]]; then
  echo "Create Python Virtual Environment on $HOSTNAME"

  python3 -m venv python_venv

  source "python_venv/bin/activate"

  pip install --upgrade pip
  pip install --upgrade pip

  pip3 install -r requirements_biomed.txt
fi


