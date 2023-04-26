#!/usr/bin/env bash

if [ "$#" -ne 3 ]; then
  echo "USAGE: $0 <max children> <seed begin> <seed end>"
  exit 1
fi

python3 generate-r-likelihoods-for-sim.py $1 $2 $3 > likelihoods.cpp
rm -rf addsim*
Rscript generate-addsim-package.r
R CMD build addsim
R CMD INSTALL addsim*.tar.gz
