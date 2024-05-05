#!/bin/bash

cd data

echo Downloading ...
wget -O newer_college.tar.gz -c https://uni-bonn.sciebo.de/s/I1YogIUalxMh8wF/download

echo Extracting dataset...
tar -xvf newer_college.tar.gz

rm newer_college.tar.gz

cd ../..
