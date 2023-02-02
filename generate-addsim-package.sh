#!/usr/bin/env bash

#module load python3.gnu/3.7.3
#module load R/3.5.1-foss-2018b

python3 generate-r-likelihoods.py > likelihoods.cpp
rm -rf addsim*
Rscript generate-addsim-package.r
R CMD build addsim
R CMD INSTALL addsim*.tar.gz
