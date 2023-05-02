import datetime
from sympy import *
from sympy.utilities.autowrap import autowrap
import numpy as np
import math
import sys

def gen_mat(m):
    if m == 1:
        return np.array([[1, 0]], dtype=np.uint8)

    Ap = gen_mat(m - 1)

    A = np.empty((m, 2**m), dtype=np.uint8)

    for i in range(m - 1):
        A[i, ::2] = Ap[i, :]
        A[i, 1::2] = Ap[i, :]

    A[-1, :2**(m - 1)] = Ap[-1, :]
    A[-1, 2**(m - 1):] = Ap[-1, :]

    return A

def gen_P(fam_size):
    num_children = fam_size - 2
    n = 2**num_children

    P_f = np.concatenate((
        np.ones((1, n), dtype=np.uint8),
        np.zeros((1, n), dtype=np.uint8),
        gen_mat(num_children)
    ))
    P_m = np.concatenate((
        np.zeros((1, n), dtype=np.uint8),
        np.ones((1, n), dtype=np.uint8),
        gen_mat(num_children)
    ))

    P = np.hstack([P_f, P_m])

    return P

class ExpressionGenerator:
    def __init__(self, max_children):
        self.family_size = max_children + 2
        self.P = gen_P(self.family_size)

        a = symbols('a')
        b = symbols('b')
        c = symbols('c')
        var_g = Symbol('var_g', positive = true)
        var_e = Symbol('var_e', positive = true)
        self.beta_0 = symbols('beta_0')
        self.beta_1 = symbols('beta_1')
        self.beta_2 = symbols('beta_2')
        self.k = Symbol('k', positive = true)

        family_size = self.P.shape[0]
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

        return exp(- genetic_frailty - shared_frailty)

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


    def print_ccode(self):
        n = self.family_size
        t = [Symbol('t_%d' % i) for i in range(1, n+1)]
        b = [Symbol('b_%d' % i) for i in range(1, n+1)]
        g = [Symbol('g_%d' % i) for i in range(1, n+1)]

        a = self.S(t, b, g)

        a_variants = [
          [
            a,
            diff(a, t[2]),
            diff(diff(a, t[2]), t[3]),
            diff(diff(diff(a, t[2]), t[3]), t[4]),
            diff(diff(diff(diff(a, t[2]), t[3]), t[4]), t[5]),
          ],
          [
            diff(a, t[0]),
            diff(diff(a, t[0]), t[2]),
            diff(diff(diff(a, t[0]), t[2]), t[3]),
            diff(diff(diff(diff(a, t[0]), t[2]), t[3]), t[4]),
            diff(diff(diff(diff(diff(a, t[0]), t[2]), t[3]), t[4]), t[5]),
          ],
          [
            diff(diff(a, t[0]), t[1]),
            diff(diff(diff(a, t[0]), t[1]), t[2]),
            diff(diff(diff(diff(a, t[0]), t[1]), t[2]), t[3]),
            diff(diff(diff(diff(diff(a, t[0]), t[1]), t[2]), t[3]), t[4]),
          ]
        ]

        args = '\n' \
        + \
        ''.join('  const double t_%d,\n' % i for i in range(1, n+1)) \
        + \
        ''.join('  const double b_%d,\n' % i for i in range(1, n+1)) \
        + \
        ''.join('  const double g_%d,\n' % i for i in range(1, n+1)) \
        + \
  '''  const double var_e,
  const double var_g,
  const double beta_0,
  const double beta_1,
  const double beta_2,
  const double k
'''
        arg_names = \
        ''.join('t_%d, ' % i for i in range(1, n+1)) \
        + \
        ''.join('b_%d, ' % i for i in range(1, n+1)) \
        + \
        ''.join('g_%d, ' % i for i in range(1, n+1)) \
        + \
        'var_e, var_g, beta_0, beta_1, beta_2, k'

        print('''#include <cmath>
#include <vector>

namespace
{
using namespace std;
''')
        m = len(a_variants)
        n = max((len(variant) for variant in a_variants))

        for i, variants in zip(range(len(a_variants)), a_variants):
            for j, l in zip(range(len(variants)), variants):
                print('''
double likelihood_%d_%d(%s) {
  return %s;
}
''' % (i, j, args, ccode(l))
                )

        variant_signatures = ('std::vector<Likelihood>{%s}' % ', '.join(('likelihood_%d_%d' % (i, j) for j in range(len(variant)))) for i, variant in zip(range(len(a_variants)), a_variants))

        print('''
using Likelihood = decltype(&likelihood_0_0);

const std::vector<std::vector<Likelihood>> likelihoods = {
  %s
};
}''' % ',\n  '.join(variant_signatures))

        print('''
// [[Rcpp::export]]
double likelihood(
  const int i,
  const int j,%s) {
  return likelihoods.at(i).at(j)(%s);
}
''' % (args, arg_names))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'USAGE: {sys.argv[0]} <max children>')
        sys.exit(1)

    max_children = int(sys.argv[1])

    generator = ExpressionGenerator(max_children)
    generator.print_ccode()
    getVariants()
