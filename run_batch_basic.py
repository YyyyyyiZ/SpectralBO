import os
import subprocess

import sys

print(sys.executable)       # path of python.exe

function_config = {
    # Synthetic Test Functions
    'branin2': {
        'n_iter': 15,
    },
    'hartmann3': {
        'n_iter': 30,
    },
    'hartmann6': {
        'n_iter': 80,
    },
    'exp10': {
        'n_iter': 150,
    },

    # Real World Datasets
    'robot3': {
        'n_iter': 100
    },
    'robot4': {
        'n_iter': 150
    },
}

n_init = 5
seed = 20
kernel_list = ['matern', 'rbf', 'rq']
acq_list = ['ei']


for func, config in function_config.items():
    n_iter = config['n_iter']
    for acq in acq_list:
        for kernel in kernel_list:
            command = (
                f"python run.py "         # change this line according to the path of python.exe
                f"--function_name {func} "
                f"--kernel {kernel} "
                f"--acq {acq} "
                f"--n_iter {n_iter} "
                f"--n_init {n_init} "
                f"--seed {seed} "
            )
            print(f"Running: {command}")
            try:
                subprocess.run(command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Command failed: {command}")
                print(f"Error: {e}")
