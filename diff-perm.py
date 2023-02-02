from sympy import *
import numpy as np
import math
import itertools 
  
n = 12
t = [Symbol('t_%d' % i) for i in range(1, n+1)]
index = [*range(0, n+1)]

#for comb in itertools.combinations(index, 3):
#    print(comb)

def part_deriv(expr, args):
    for i in args:
        expr = diff(expr, t[i])

    return expr


s = exp(sum(ti**2 for ti in t))

pprint(part_deriv(s,(0, 1, 2)))
