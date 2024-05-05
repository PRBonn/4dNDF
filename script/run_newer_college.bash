#!/bin/bash

python static_mapping.py config/surface/newer_college.yaml

cd eval

## crop the reconstructed mesh based on the reference
python mesh_crop.py

## evaluate the croped mesh
python eval_newercollege.py
