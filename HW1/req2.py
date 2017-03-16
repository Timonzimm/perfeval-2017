import pycurl
from urllib.parse import urlencode
from io import BytesIO
import AdvancedHTMLParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import setp
import copy
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARMA 
import matplotlib.ticker as plticker

parser = AdvancedHTMLParser.AdvancedHTMLParser()
c = pycurl.Curl()
c.setopt(c.URL, 'http://tcpip.epfl.ch/perfeval-lab1/output.php')

paramBase = {
    'sciper': 223720,
    'clients': 1,
    'apoints': 1,
    'servers': 1
}

columns = [
    'Successful download requests per second',
    'Packets per second', 
    'Collision probability',
    'Delay'
]


def query(param):
    postfields = urlencode(param)
    c.setopt(c.POSTFIELDS, postfields)
    c.perform()
    buffer = BytesIO()
    c.setopt(c.WRITEDATA, buffer)
    c.perform()
    body = buffer.getvalue()
    body = body.decode('iso-8859-1')
    parser.parseStr(body)
    t = parser.getElementsByTagName('td')
    t = [x.innerHTML for x in t]
    return [float(t[11]), float(t[13]), float(t[15]), float(t[17])]

def sameReq(nbReq, param):
    results = []
    for i in range(nbReq):
        results.append(query(param))
    return np.asarray(results)


def diffReq(nbReq, param, field, step):
    results = []
    p = list(range(0, nbReq*step, step))
    p[0] = 1
    for i in range(nbReq):
        param[field] = p[i]
        results.append(query(param))
    return np.asarray(results)

def getLinearParam(c):
    inters = [(1,55), (55,115), (115,175), (175, 235), (235,305), (305,355), (355,425), (425,475), (475,545), (545,595), (595,10000)]
    params = [(1,1), (2,1), (3,1), (4,1), (5,1), (6,2), (7,2), (8,2), (9,2), (10,2), (10,3)]

    for i in range(11):
        if (c >= inters[i][0] and c <= inters[i][1]):
            return params[i]
    return (1,1)


def linearReq(nbReq, param, field, step):
    results = []
    p = list(range(0, nbReq*step, step))
    p[0] = 1
    for i in range(nbReq):
        param[field] = p[i]
        a, s =getLinearParam(i)
        param['apoints'] = a
        param['servers'] = s
        results.append(query(param))
    return np.asarray(results)

def mean(nbMean, nbReq, param, field, step):
    results = []
    for i in range(nbMean):
        results.append(diffReq(nbReq, param, field, step))
        print(i)
    return np.asarray(results)

def meanLinear(nbMean, nbReq, param, field, step):
    results = []
    for i in range(nbMean):
        results.append(linearReq(nbReq, param, field, step))
        print(i)
    return np.asarray(results)    

def movingaverage(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    res = np.apply_along_axis(lambda m: np.convolve(m, window, 'same'), axis=0, arr=data)
    return res


def fetchDataQ1():
    res = sameReq(1000, paramBase)
    np.save('./Q1/sameReq1000', res)

def fetchDataQ2():
    res = mean(10, 1001, paramBase, 'clients', 1)
    np.save('./Q2/diffReq10x1000', res)

def fetchDataQ3():
    params = [copy.deepcopy(paramBase),
    copy.deepcopy(paramBase),
    copy.deepcopy(paramBase),
    copy.deepcopy(paramBase)]

    params[1]['apoints'] = 2 * params[0]['apoints']
    params[2]['apoints'] = 2 * params[1]['apoints']
    params[3]['apoints'] = 2 * params[2]['apoints']

    res = []

    for p in params:
        res.append(mean(10, 1001, p, 'clients', 1))
    
    res = np.asarray(res)

    np.save('./Q3/data', res)

def fetchDataQ4():
    res = []
    print(np.shape(res))
    for a in range (10):
        for s in range(10):
            print(a,s)
            p = copy.deepcopy(paramBase)
            p['apoints'] = a + 1
            p['servers'] = s + 1
            res.append(diffReq(1001, p, 'clients', 1))
    np.save('./Q4/data', np.asarray(res))

def fetchDataLinear():
    res = meanLinear(10, 1001, paramBase, 'clients', 1)
    np.save('./Q4/linear', res)



def plotSeries(data, labels, path, x, y, title='Test', vline=None):
    loc = plticker.MultipleLocator(base=200.0)
    df = pd.DataFrame(data, columns=labels)
    ax = df.plot(lw=0.5,colormap='jet',marker='.',markersize=0,title=title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)    plt.savefig(path, format='png', dpi=400)
    ax.grid()
    ax.xaxis.set_major_locator(loc)
    if (vline):
        plt.axvline(x=vline)
        plt.text(vline, 0, '%d clients' % vline, rotation=90, verticalalignment='bottom')
        ax.legend(loc='upper left', labelspacing=0)
    '''for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(3) 
    for label in ax.xaxis.get_ticklabels()[::1]:
        label.set_visible(False)
    for label in ax.xaxis.get_ticklabels()[::5]:
        label.set_visible(True)'''

    plt.savefig(path, format='png', dpi=400)

def plotBox(data, labels, path, title='Test'):
    df = pd.DataFrame(data, columns=[labels])
    ax = df.plot.box(colormap='jet',title=title)
    ax.grid()
    plt.savefig(path, format='png', dpi=400)

def plotSwag(mean, std, labels, path, x, y, title='Test'):
    trials = np.array(range(len(mean))) + 1
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid()
    ax.fill_between(trials, mean - std, mean + std, alpha=0.1, color="g")
    ax.plot(trials, mean, '-', color="g", linewidth=1)
    plt.savefig(path, format='png', dpi=400)

def plotQ1():
    res = np.load('./Q1/data.npy')
    print(res.shape)
    for i in range(4):
        plotBox(res[:,i], columns[i], './Q1/{0}.png'.format(columns[i]), title='')
    

def plotQ2():
    res = np.load('./Q2/data.npy')
    '''res = res[0]
    res = res.transpose()
    print(res.shape)
    for i in range(4):
        plotSeries(res[i], [columns[i]], 
        './Q2/{0}.svg'.format(columns[i]), x='Number of clients', y=columns[i])'''
    mean = res.mean(0).transpose()
    std = res.std(0).transpose()
    for i in range(4):
        plotSwag(mean[i], std[i], [columns[i]], 
        './Q2/{0}.png'.format(columns[i]), x='Number of request per second', y=columns[i], title='')





def plotQ3():
    res = np.load('./Q3/data.npy')
    print(np.shape(res))
    resMean = np.mean(res, 1)
    resMean = np.asarray([res[0][0], res[1][0], res[2][0], res[3][0]])
    print(np.shape(resMean))
    resMean = np.transpose(resMean, (0,2,1))

    def extract(dim):
        res = []
        for x in range(4):
            res.append(resMean[x][dim])
        return np.asarray(res)
    
    thetas = extract(0)
    pps = extract(1)
    cp = extract(2)
    d = extract(3)
    plotSeries(np.asarray(thetas).transpose(), ['1 apoint','2 apoints','4 apoints','8 apoints'], './Q3/{0}.png'.format(columns[0]), x='Number of request per second', y=columns[0], title='')
    plotSeries(np.asarray(pps).transpose(), ['1 apoint','2 apoints','4 apoints','8 apoints'], './Q3/{0}.png'.format(columns[1]), x='Number of request per second', y=columns[1], title='')
    plotSeries(np.asarray(cp).transpose(), ['1 apoint','2 apoints','4 apoints','8 apoints'], './Q3/{0}.png'.format(columns[2]), x='Number of request per second', y=columns[2], title='')
    plotSeries(np.asarray(d).transpose(), ['1 apoint','2 apoints','4 apoints','8 apoints'], './Q3/{0}.png'.format(columns[3]), x='Number of request per second', y=columns[3], title='')

def plotQ4():
    res = np.load('./Q4/data.npy')
    for a in range(100):
        p = res[a]
        plotSeries(p.transpose()[0], [columns[0]], 
        './Q4/S{0}A{1}.png'.format(a%10 + 1,a/10 + 1), x='Number of request per second', y=columns[0])

def plotQ4alt():
    res = np.load('./Q4/data.npy')
    res = res.transpose((2,1,0))
    res = res[0]
    vline = [55, 115, 175, 235, 305, 355, 425, 475, 545, 595]
    print(res.shape)
    for i in range(10):
        labs = list(range(1,11,1))
        labs = ['{0}AP,{1}S '.format(i+1, x) for x in labs]
        print(res[:, i*10:i*10 + 10].shape)
        p = res[:, i*10:i*10 + 10]
        plotSeries(p, labs, 
        './Q4/test{0}.png'.format(i), x='Number of request per second', y=columns[0], vline=vline[i], title='')

def plotLinear():
    res = np.load('./Q4/linear.npy')
    mean = res.mean(0).transpose()
    std = res.std(0).transpose()
    for i in range(4):
        plotSwag(mean[i], std[i], [columns[i]], 
        './Q4/{0}.png'.format(columns[i]), x='Number of request per second', y=columns[i], title='')

def plotFinal():
    inters = [(1,55), (55,115), (115,175), (175, 235), (235,305), (305,355), (355,425), (425,475), (475,545), (545,595), (595,10000)]
    params = [(1,1), (2,1), (3,1), (4,1), (5,1), (6,2), (7,2), (8,2), (9,2), (10,2), (10,3)]

    inters = np.asarray([x[0] for x in inters]).reshape(len(params), 1)
    a = np.asarray([x[0] for x in params]).reshape(len(params), 1)
    s = np.asarray([x[1] for x in params]).reshape(len(params), 1)

    X = np.asarray(range(1, 601, 1)).reshape(600,1)

    reg = LinearRegression() 
    reg.fit(inters, a) 


    df1 = pd.DataFrame(a,)
    ax = df1.plot(lw=1,colormap='jet',marker='.',markersize=6, x=inters, )
    ax.set_xlabel('Number of requests per second')
    ax.set_ylabel('Access points number')


    ax.grid()
    plt.savefig('./Q4/apoints.png', format='png', dpi=400)








plotFinal()