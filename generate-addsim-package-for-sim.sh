#!/usr/bin/env bash

if [ "$#" -ne 3 ]; then
  echo "USAGE: $0 <max children> <seed begin> <seed end>"
  exit 1
fi

python3 generate-r-likelihoods-for-sim.py $1 $2 $3 > likelihoods.cpp
rm -rf addsim$1*
Rscript generate-addsim-package.r $1
R CMD build addsim$1
R CMD INSTALL addsim$1*.tar.gz
