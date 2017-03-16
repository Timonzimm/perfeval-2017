import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def Q1():
    sampNumb = np.arange(0,9,1)
    sampNumb = np.array([10 * (3**x) for x in sampNumb])
    
    samples = [np.random.normal(size=x) for x in sampNumb]
    
    #hists = [np.histogram(x, bins=10) for x in samples]

    print(type(samples[0]))

    for i in range(len(samples)):
        df = pd.DataFrame(samples[i])
        df.plot.hist(bins=100)
        plt.savefig('./Q1/hist{0}.png'.format(sampNumb[i]), format='png', dpi=400)

def wp_sim(width, sciper, end_time):
    v_min = 0.7 + 0.01 * (sciper%21)
    v_max = 1.9 + 0.1 * (sciper%11)

    M = np.random.uniform(0, width, 2)
    V = np.random.uniform(v_min, v_max, 1)[0]
    T = 0

    while True:
        if (T >= end_time):
            return
        yield M
        M_t = np.random.uniform(0, width, 2)
        T = T + (np.linalg.norm(M - M_t) / V)
        M = M_t
        V = np.random.uniform(v_min, v_max, 1)[0]



def wp_n_sim(N):
    sims = []
    
    for i in range(N):
        tmp = []
        wp = wp_sim(1500, 226977, 86000)
        for m in wp:
            tmp.append(m)
        sims.append(np.array(tmp))
    
    return sims


def Q2():
    sims = wp_n_sim(100)
    mean = [x.shape[0] for x in sims]
    print(np.mean(mean))

    '''for i in range(10):
        x = sims[i][:,0]
        y = sims[i][:,1]
        c = np.arange(x.size) / x.size

        plt.scatter(x, y, c=c, cmap='Greens')
        plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, width=0.001)

        plt.savefig('./Q2/{0}.png'.format(i), format='png', dpi=400)
        plt.close()'''

Q2()

    