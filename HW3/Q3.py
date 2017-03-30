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

    tests = []
    for i in range(len(changes_1)):
        idx = np.searchsorted(times[i], np.arange(times[0][-1]))
        idx = np.clip(idx, 1, len(changes_1[i])-1) 
        t = np.cumsum(changes_1[i])[idx]
        tests.append(t)

    m = np.mean(tests, 0)
'''    print('cov')
    t = np.convolve(m, np.ones(50)*(1/50))'''
    
    plt.plot( )
    plt.show()

    plt.plot(times[0])
    plt.show()

    plt.plot(np.cumsum(changes_1[0]))
    plt.show()

    plt.plot(np.cumsum(changes_1[1]))
    plt.show()

    mean_changes_1 = [np.mean(np.cumsum(x)) for x in changes_1]
    mean_changes_2 = [np.mean(np.cumsum(x)) for x in changes_2]
    
    print('Averages of type 1 in system: {}'.format(mean_changes_1))
    print('Averages of type 2 in system: {}'.format(mean_changes_2))



