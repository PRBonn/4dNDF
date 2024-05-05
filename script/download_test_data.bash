#!/bin/bash

cd data

echo Downloading 20 frames from KITTI seq.00 ...
wget -O test.tar.gz -c https://uni-bonn.sciebo.de/s/X8hVgZQGYl2WTPk/download

echo Extracting dataset...
tar -xvf test.tar.gz

rm test.tar.gz

cd ../..
