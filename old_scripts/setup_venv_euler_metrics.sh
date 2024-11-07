#!/bin/bash

#module load  gcc/6.3.0 python_gpu/3.7.4
#module load gcc/8.2.0 python_gpu/3.9.9 cuda/11.3.1
module load gcc/8.2.0 python_gpu/3.9.9 cuda/12.1.1

if [[ ! -d "metrics_python_venv" ]]; then
  echo "Create Python Virtual Environment on $HOSTNAME"

  python3 -m venv metrics_python_venv

  source "metrics_python_venv/bin/activate"

  pip install --upgrade pip
  pip install --upgrade pip

  pip3 install -r requirements_euler_metrics.txt
fi
