#!/bin/bash

#module load  gcc/6.3.0 python_gpu/3.7.4
#module load gcc/8.2.0 python_gpu/3.9.9
module load stack/.2024-05-silent stack/2024-05
module load gcc/13.2.0
module load cuda/12.2.1
module load cudnn/9.2.0
module --ignore_cache load python/3.9.18
# module load python_cuda/3.9.18

if [[ ! -d "python_venv" ]]; then
  echo "Create Python Virtual Environment on $HOSTNAME"

  python3 -m venv python_venv

  source "python_venv/bin/activate"

  pip install --upgrade pip
  pip install --upgrade pip

  # pip3 install -r requirements_euler.txt
  pip3 install -r new_requirements.txt
fi
