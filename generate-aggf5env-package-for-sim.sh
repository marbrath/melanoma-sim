#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
  echo "USAGE: $0 <max children>"
  exit 1
fi

python3 generate-aggf5env-r-likelihoods-for-sim.py $1 > likelihoods.cpp
rm -rf aggf5env*
Rscript generate-aggf5env-package.r
R CMD build aggf5env
R CMD INSTALL aggf5env*.tar.gz
