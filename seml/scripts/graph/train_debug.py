#!/usr/bin/env python3
import yaml
import os
from train import run

if __name__ == '__main__':
    CUDA_LAUNCH_BLOCKING = 1
    with open("seml/configs/graph/train_ps.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    grid_params = {}
    for k in config['grid']:
        if config['grid'][k]['type'] == 'choice':
            grid_params[k] = config['grid'][k]['options'][0]
        elif config['grid'][k]['type'] == 'range':
            grid_params[k] = config['grid'][k]['min']
    run(**config['fixed'], **grid_params)
