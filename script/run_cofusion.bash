#!/bin/bash


python static_mapping.py config/surface/cofusion.yaml

cd eval

python eval_cofusion.py
