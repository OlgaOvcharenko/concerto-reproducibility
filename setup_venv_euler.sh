#!/bin/bash

#module load  gcc/6.3.0 python_gpu/3.7.4
module load gcc/8.2.0 python_gpu/3.9.9

if [[ ! -d "python_venv" ]]; then
  echo "Create Python Virtual Environment on $HOSTNAME"

  python3 -m venv python_venv

  source "python_venv/bin/activate"

  pip install --upgrade pip
  pip install --upgrade pip

  pip3 install -r requirements_euler.txt
fi
