import datetime
from sympy import *
from sympy.utilities.autowrap import autowrap
from permutation_fast import get_parent_matrices
import numpy as np
import itertools
import os
import sys


class ExpressionGenerator:
    def __init__(self, max_children):
        self.family_size = 2 + max_children
        self.max_num_sick = 5

        P_f, P_m = get_parent_matrices(self.family_size)
        P_f = P_f[:self.family_size]
        P_m = P_m[:self.family_size]

        self.P = np.hstack([P_f, P_m])

        var_g = Symbol('var_g', positive=True)
        var_e = Symbol('var_e', positive=True)
        self.beta_0 = symbols('beta_0', real=True)
        self.beta_1 = symbols('beta_1', real=True)
        self.beta_2 = symbols('beta_2', real=True)
        self.k = Symbol('k', positive=True)

        num_genes = self.P.shape[1]

        var_sum = var_e + var_g
        self.eta = 1/var_sum
        self.nu_e = var_e/var_sum**2
        nu_g = var_g/var_sum**2
        self.nu_gi = nu_g/(num_genes/2)

        self.v_phi_1 = np.vectorize(self.phi_1)
        self.v_phi_2 = np.vectorize(self.phi_2)
        self.v_cum_haz = np.vectorize(self.cum_haz)
        self.v_regr = np.vectorize(self.regr)

    def phi_1(self, c):
        return self.nu_gi*(log(self.eta + c) - log(self.eta))

    def phi_2(self, c):
        return self.nu_e*(log(self.eta + c) - log(self.eta))

    def cum_haz(self, c):
        return c**self.k

    def regr(self, a, b):
        return exp(self.beta_0 + self.beta_1*a + self.beta_2*b)

    def Laplace(self, lifetimes):
      genetic_frailty = sum(self.v_phi_1(np.dot(lifetimes, self.P)))
      shared_frailty = self.v_phi_2(np.dot(lifetimes, np.ones(self.family_size)))

      return exp(-genetic_frailty - shared_frailty)

    def S(self, lifetimes, birthyears, genders):
      return self.Laplace(np.multiply(self.v_regr(birthyears, genders), self.v_cum_haz(lifetimes)))

    def to_fast_func(self, expr, dont_optimize, args):
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

    def part_deriv(self, expr, args):
        for arg in args:
            expr = diff(expr, arg)

        return expr

    def print_ccode(self, sick_ids):
        n = self.family_size
        t = [Symbol('t_%d' % i) for i in range(1, n+1)]
        b = [Symbol('b_%d' % i) for i in range(1, n+1)]
        g = [Symbol('g_%d' % i) for i in range(1, n+1)]

        a = self.S(t, b, g)

        indices = range(len(t))
        all_combs = itertools.chain(*(itertools.combinations(indices, i) for i in range(self.max_num_sick)))
        a_variants = {0: a}

        for num_sick_parents in range(min(2, self.max_num_sick) + 1):
            for num_sick_children in range(0, self.max_num_sick - num_sick_parents + 1):_
                comb = list(range(num_sick_parents)) + list(range(2, num_sick_children + 2))
                id = sum(1 << i for i in comb)

                sick_ids.add(id)


        for comb in all_combs:
            id = sum(1 << i for i in comb)

            if id not in sick_ids:
                continue

            a_variants[id] = self.part_deriv(a, (t[i] for i in comb))

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


def read_sick_ids(seeds):
    all_sick_ids = set()

    for seed in seeds:
        path = os.path.join('sim-output', f'npy_files_{seed:04d}_{max_children:02d}', 'sick_ids.npy')
        sick_ids = np.load(path)
        all_sick_ids = all_sick_ids.union({*sick_ids})

    return all_sick_ids


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'USAGE: {sys.argv[0]} <max children> <seed begin> <seed end>')
        sys.exit(1)

    max_children = int(sys.argv[1])
    seed_begin = int(sys.argv[2])
    seed_end = int(sys.argv[3])

    sick_ids = read_sick_ids(range(seed_begin, seed_end))

    generator = ExpressionGenerator(max_children)
    generator.print_ccode(sick_ids)
