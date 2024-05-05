#!/bin/bash

cd data

echo Downloading ...
wget -O baseline.tar.gz -c https://uni-bonn.sciebo.de/s/SibmxR6NaoeRffc/download

echo Extracting dataset...
tar -xvf baseline.tar.gz

rm baseline.tar.gz

cd ../..
