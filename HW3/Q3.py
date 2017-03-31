import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simulator import Simulator
from math import *


def ci_lambda(l, n):
    sim = Simulator(lambda_=l)
    res = sim.simulate(num_run=n)

    res = [sorted(x[1], key=lambda tup: tup[0]) for x in res]

    times = [[x[0] for x in y] for y in res]
    changes_1 = [[x[2] for x in y] for y in res]
    changes_2 = [[x[3] for x in y] for y in res]

    samples = np.hstack(x for x in times)
    samples = np.sort(samples)

    tests = []
    for i in range(len(changes_1)):
        idx = np.searchsorted(times[i], samples)
        idx = np.clip(idx, 1, len(changes_1[i])-1) 
        t = np.cumsum(changes_1[i])[idx]
        tests.append(t)

    m = np.mean(tests, 0)

    np.savetxt('n50_l108', m)
    np.savetxt('n50_l108_samples', samples)


def ci_med(serie):
    n = serie.size
    j = floor( 0.50*n - 0.980*sqrt(n))
    k = ceil(0.50*n + 0.980*sqrt(n) + 1)
    return (serie[j],  serie[k])

def ci_mean(serie):
    n = serie.size
    m = serie.mean()
    s = np.std(serie)
    return (m - 1.96*s/sqrt(n), m + 1.96*s/sqrt(n))

    


def ci():
    l70 = np.loadtxt('n50_l70.gz')
    l108 = np.loadtxt('n50_l108.gz')

    ci70_med = ci_med(l70)
    ci70_mean = ci_mean(l70)
    ci108_med = ci_med(l108)
    ci108_mean = ci_mean(l108)

    print('Confidence inverval for the mediane at lambda = 70: {}', ci70_med)
    print('Confidence inverval for the mean at lambda = 70: {}', ci70_mean)
    print('Confidence inverval for the mediane at lambda = 108: {}', ci108_med)
    print('Confidence inverval for the mediane at lambda = 108: {}', ci108_mean)

ci()
