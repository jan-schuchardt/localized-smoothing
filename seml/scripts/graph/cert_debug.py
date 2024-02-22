#!/usr/bin/env python3
import yaml
import os
from cert import run

if __name__ == '__main__':
    with open("seml/configs/graph/cert_appnp_seeds05065_DEBUG.yaml",
              "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    run(**config['fixed'])
