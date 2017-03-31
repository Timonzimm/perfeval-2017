import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simulator import Simulator


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

    np.savetxt('n50_l70', m)
    np.savetxt('n50_l70_samples', samples)

ci_lambda(70, 50)





