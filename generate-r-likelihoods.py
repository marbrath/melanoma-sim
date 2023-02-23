import datetime
from sympy import *
from sympy.utilities.autowrap import autowrap
from permutation_fast import get_parent_matrices
import numpy as np
import math
import itertools

num_children = 8
family_size = 2 + num_children
max_num_sick = 5

P_f, P_m = get_parent_matrices(family_size)
P_f = P_f[:family_size]
P_m = P_m[:family_size]

P = np.hstack([P_f, P_m])

a = symbols('a')
b = symbols('b')
c = symbols('c')
var_g = Symbol('var_g', positive = true)
var_e = Symbol('var_e', positive = true)
beta_0 = symbols('beta_0')
beta_1 = symbols('beta_1')
beta_2 = symbols('beta_2')
k = Symbol('k', positive = true)

family_size = P.shape[0]
num_genes = P.shape[1]

var_sum = var_e + var_g
eta = 1/var_sum
nu_e = var_e/var_sum**2
nu_g = var_g/var_sum**2
nu_gi = nu_g/(num_genes/2)

def phi_1(c):
 	return nu_gi*(log(eta + c) - log(eta))

def phi_2(c):
    return nu_e*(log(eta + c) - log(eta))

def cum_haz(c):
    return c**k

def regr(a, b):
    return exp(beta_0 + beta_1*a + beta_2*b)

v_phi_1 = np.vectorize(phi_1)
v_phi_2 = np.vectorize(phi_2)
v_cum_haz = np.vectorize(cum_haz)
v_regr = np.vectorize(regr)

def Laplace(lifetimes):
  genetic_frailty = sum(v_phi_1(np.dot(lifetimes, P)))
  shared_frailty = v_phi_2(np.dot(lifetimes, np.ones(family_size)))

  return exp(- genetic_frailty - shared_frailty) 

def S(lifetimes, birthyears, genders):
  return Laplace(np.multiply(v_regr(birthyears, genders), v_cum_haz(lifetimes)))

def to_fast_func(expr, dont_optimize, args):
  begin = datetime.datetime.now()
  func = autowrap(
              args=args,
              expr=expr,
              backend='cython',
              tempdir='objects',
              extra_compile_args=['-O0'] if dont_optimize else []
          )
  end = datetime.datetime.now()
  print('Autowrap took %fs' % (end - begin).total_seconds())

  return func

def part_deriv(expr, args):
    for arg in args:
        expr = diff(expr, arg)

    return expr


def getVariants():
  n = family_size
  t = [Symbol('t_%d' % i) for i in range(1, n+1)]
  b = [Symbol('b_%d' % i) for i in range(1, n+1)]
  g = [Symbol('g_%d' % i) for i in range(1, n+1)]

  args = t + b + g + [var_e, var_g, beta_0, beta_1, beta_2, k]

  a = S(t, b, g)

  indices = range(len(t))
  all_combs = itertools.chain(*(itertools.combinations(indices, i) for i in range(max_num_sick)))
  a_variants = {0: a}

  for comb in all_combs:
      id = sum(1 << i for i in comb)
      a_variants[id] = part_deriv(a, (t[i] for i in comb))

  param_args = '''  const double var_e,
  const double var_g,
  const double beta_0,
  const double beta_1,
  const double beta_2,
  const double k
'''

  args = '\n' \
  + \
  ''.join('  const double t_%d,\n' % i for i in range(1, n+1)) \
  + \
  ''.join('  const double b_%d,\n' % i for i in range(1, n+1)) \
  + \
  ''.join('  const double g_%d,\n' % i for i in range(1, n+1)) \
  + \
  param_args

  arg_names = \
  ''.join('t[%d], ' % i for i in range(n)) \
  + \
  ''.join('b[%d], ' % i for i in range(n)) \
  + \
  ''.join('g[%d], ' % i for i in range(n)) \
  + \
  'var_e, var_g, beta_0, beta_1, beta_2, k'

  print('''#include "Rcpp.h"

#include <cmath>
#include <unordered_map>

namespace
{
using namespace std;
''')

  for id in a_variants:
    print(
  '''
double likelihood_%d(%s) {
  return %s;
}
''' % (id, args, ccode(a_variants[id]))
      )

  variant_signatures = ('{%d, likelihood_%d}' % (id, id) for id in a_variants)

  print('''
using Likelihood = decltype(&likelihood_0);

const std::unordered_map<uint32_t, Likelihood> likelihoods = {
  %s
};
}''' % ',\n  '.join(variant_signatures))

  print('''
// [[Rcpp::export]]
double likelihood(
  const int sick_id,
  const Rcpp::NumericVector& t,
  const Rcpp::NumericVector& b,
  const Rcpp::NumericVector& g,
%s
) {
  return likelihoods.at(sick_id)(%s);
}
''' % (param_args, arg_names))

if __name__ == '__main__':
  getVariants()
