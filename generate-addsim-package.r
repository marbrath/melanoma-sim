library('Rcpp')

args = commandArgs(trailingOnly=TRUE)
if (length(args) != 1) {
  stop("Usage: generate-addsim-package.r <max children>")
}

max_children = as.integer(args[1])

Rcpp.package.skeleton(sprintf('addsim%d', max_children), attributes=TRUE, example_code=FALSE, cpp_files=c('likelihoods.cpp'))
