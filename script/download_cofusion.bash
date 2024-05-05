#!/bin/bash

cd data

echo Downloading ...
wget -O cofusion.tar.gz -c https://uni-bonn.sciebo.de/s/ziThN7VSgvNHOxc/download

echo Extracting dataset...
tar -xvf cofusion.tar.gz

rm cofusion.tar.gz

cd ../..
