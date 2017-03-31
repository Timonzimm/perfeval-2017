import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import bisect
import scipy as sp
import scipy.stats



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
    V = np.random.uniform(v_min, v_max)
    T = 0

    while True:
        if (T >= end_time):
            return
        yield (M, V, T)
        M_t = np.random.uniform(0, width, 2)
        T = T + (np.linalg.norm(M - M_t) / V)
        M = M_t
        V = np.random.uniform(v_min, v_max)



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
    sims = [ [y[0] for y in x] for x in sims]
    sims = [np.array(x) for x in sims]

    waypoints_numbers = [x.shape[0] for x in sims]
    max_ = np.max(waypoints_numbers)
    min_ = np.min(waypoints_numbers)
    mean = np.mean(waypoints_numbers)

    print(min_, max_, mean)


    COLOR='blue'
    RESFACT=10
    MAP='winter' # choose carefully, or color transitions will not appear smoooth

    sim = sims[0]
    x = sim[:,0]
    y = sim[:,1]

    cm = plt.get_cmap(MAP)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/(len(x)-1)) for i in range(len(x)-1)])
    for i in range(len(x)-1):
      ax.plot(x[i:i+2],y[i:i+2])
    ax.scatter(x, y, s=50, facecolor='none', edgecolor='black')  
    ax.scatter(x[0], y[0], s=200, facecolor='#0000ff', edgecolor='black', linewidth=2)
    ax.scatter(x[-1], y[-1], s=200, facecolor='#00fd80', edgecolor='black', linewidth=2)
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    

    plt.savefig('./Q2/5.png', format='png', dpi=400)
    plt.close()

    '''for i in range(10):
        x = sims[i][:,0]
        y = sims[i][:,1]
        c = np.arange(x.size) / x.size

        plt.scatter(x, y, c=c, cmap='Greens')
        plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, width=0.001)

        plt.savefig('./Q2/{0}.png'.format(i), format='png', dpi=400)
        plt.close()'''

def Q3_1():
    sims = wp_n_sim(100)

    speed = [np.array([y[1] for y in x]) for x in sims]
    m = [np.array([y[0] for y in x]) for x in sims]

    m1 = [x[0] for x in sims[0]]
    print(np.array(m1).shape)

    m_all = []
    for x in m:
        for y in x:
            m_all.append(y)

    s = np.hstack(x for x in speed)
    print(np.array(m1).shape, "--------------------------")

    df = pd.DataFrame(speed[0])
    df.plot.hist(legend=False)

    plt.xlabel('bins')
    plt.ylabel('frequency')
    
    plt.savefig('./Q3/hist_1_sim.png', format='png', dpi=400)

    # plot m 1
    df = pd.DataFrame(m1, columns=['a', 'b'])
    df.plot.hexbin(gridsize=20, x='a', y='b', legend=False, reduce_C_function=len)

    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    
    plt.savefig('./Q3/evet_hist1_m.png', format='png', dpi=400)

    # plot m all
    df = pd.DataFrame(m_all, columns=['a', 'b'])
    df.plot.hexbin(gridsize=20, x='a', y='b', legend=False, reduce_C_function=len)

    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    
    plt.savefig('./Q3/event_histall_m.png', format='png', dpi=400)    




def time_sampling(s):

    m = np.array([x[0] for x in s])
    speed = np.array([x[1] for x in s])
    time = np.array([x[2] for x in s])

    intervals = np.arange(int(time[-1]/12)) * 12


    res = []


    for i in intervals:
        i_left = bisect.bisect_left(time, i)
        i_right = i_left + 1

        t1 = time[i_left]
        if (i_left != len(time) -1):
            t2 = time[i_right]
            alpha = (t2 - i)/(t2 - t1)
        else:
            i_right = i_left
            alpha = 0
        res.append((speed[i_left], m[i_left] * (1- alpha) + m[i_right]*alpha))

    return res



def Q3_2():
    sims = wp_n_sim(100)
    sample1 = time_sampling(sims[0])
    sample_all = [time_sampling(x) for x in sims]

    

    sample1_speed = np.array([x[0] for x in sample1])
    sample1_all_speed = [np.array([y[0] for y in x]) for x in sample_all]


    sample1_m = np.array([x[1] for x in sample1])
    sample1_all_m = [np.array([y[1] for y in x]) for x in sample_all]


    speed_all = np.hstack(x for x in sample1_all_speed)
    m_all = []
    for x in sample1_all_m:
        for y in x:
            m_all.append(y)

    # plot speed 1    
    df = pd.DataFrame(sample1_speed)
    df.plot.hist(legend=False)

    plt.xlabel('bins')
    plt.ylabel('frequency')
    
    plt.savefig('./Q3/hist1_speed.png', format='png', dpi=400)


    # plot speed speed_all
    df = pd.DataFrame(speed_all)
    df.plot.hist(legend=False)

    plt.xlabel('bins')
    plt.ylabel('frequency')
    
    plt.savefig('./Q3/histall_speed.png', format='png', dpi=400)

    # plot m 1
    df = pd.DataFrame(sample1_m, columns=['a', 'b'])
    df.plot.hexbin(gridsize=20, x='a', y='b', legend=False, reduce_C_function=len)

    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    
    plt.savefig('./Q3/hist1_m.png', format='png', dpi=400)

    # plot m all
    df = pd.DataFrame(m_all, columns=['a', 'b'])
    df.plot.hexbin(gridsize=20, x='a', y='b', legend=False, reduce_C_function=len)

    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    
    plt.savefig('./Q3/histall_m.png', format='png', dpi=400)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def median_confidence_interval(data, confidence=0.95):
    w = data+1
    rv = sp.stats.rv_discrete(values=(data, w/float(w.sum())))
    (down, up) = rv.interval(0.95) 
    return rv.median(), down, up



def Q4():
    sims = wp_n_sim(25)
    sims_tsamp = [time_sampling(x) for x in sims]

    speed = [np.array([y[1] for y in x]) for x in sims]
    speed_tsamp = [np.array([y[0] for y in x]) for x in sims_tsamp]

    speed_mean_stats = [mean_confidence_interval(x) for x in speed]
    speed_tsamp_mean_stats = [mean_confidence_interval(x) for x in speed_tsamp]

    speed_mean = [x[0] for x in speed_mean_stats]
    speed_meanUp = [x[1] for x in speed_mean_stats]
    speed_meanDown = [x[2] for x in speed_mean_stats]

    speed_tsamp_mean = [x[0] for x in speed_tsamp_mean_stats]
    speed_tsampmeanUp = [x[1] for x in speed_tsamp_mean_stats]
    speed_tsampmeanDown = [x[2] for x in speed_tsamp_mean_stats]    




    speed_med_stats = [median_confidence_interval(x) for x in speed]
    speed_tsamp_med_stats = [median_confidence_interval(x) for x in speed_tsamp]

    speed_med = [x[0] for x in speed_med_stats]
    speed_medUp = [x[1] for x in speed_med_stats]
    speed_medDown = [x[2] for x in speed_med_stats]

    speed_tsamp_med = [x[0] for x in speed_tsamp_mean_stats]
    speed_tsampmedUp = [x[1] for x in speed_tsamp_mean_stats]
    speed_tsampmedDown = [x[2] for x in speed_tsamp_mean_stats]

    X = np.arange(len(speed_mean))    
    plt.scatter(x=X, y=speed_mean, s=10)
    plt.scatter(x=X, y=speed_meanUp, marker='_', c='blue')
    plt.scatter(x=X, y=speed_meanDown, marker='_', c='blue')
    plt.scatter(x=X, y=speed_med, c='red', s=10)
    plt.scatter(x=X, y=speed_medUp, marker='_', c='red')
    plt.scatter(x=X, y=speed_medDown, marker='_', c='red')
    plt.xlabel('25 simulations')
    plt.ylabel('mean, median and ci')


    plt.savefig('./Q4/mmci25event.png', format='png', dpi=400)
    plt.clf()

    X = np.arange(len(speed_tsamp_mean))    
    plt.scatter(x=X, y=speed_tsamp_mean, s=10)
    plt.scatter(x=X, y=speed_tsampmeanUp, marker='_', c='blue')
    plt.scatter(x=X, y=speed_tsampmeanDown, marker='_', c='blue')
    plt.scatter(x=X, y=speed_tsamp_med, c='red', s=10)
    plt.scatter(x=X, y=speed_tsampmedUp, marker='_', c='red')
    plt.scatter(x=X, y=speed_tsampmedDown, marker='_', c='red')
    plt.xlabel('25 simulations')
    plt.ylabel('mean, median and ci')

    plt.savefig('./Q4/mmci25time.png', format='png', dpi=400)
    plt.clf()

Q4()


