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

def mean(nbMean, nbReq, param, field, step):
    results = []
    for i in range(nbMean):
        results.append(diffReq(nbReq, param, field, step))
        print(i)
    return np.asarray(results)



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

def plotSeries(data, labels, path, x, y, title='Test'):
    df = pd.DataFrame(data, columns=labels)
    ax = df.plot(lw=0.5,colormap='jet',marker='.',markersize=0,title=title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid()
    plt.savefig(path, format='svg', dpi=800)

def plotSwag(mean, std, labels, path, x, y, title='Test'):
    trials = np.array(range(len(mean))) + 1
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Test')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid()
    ax.fill_between(trials, mean - std, mean + std, alpha=0.1, color="g")
    ax.plot(trials, mean, '-', color="g", linewidth=1)
    plt.savefig(path, format='svg', dpi=800)

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
        './Q2/{0}.svg'.format(columns[i]), x='Number of clients', y=columns[i])





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
    plotSeries(np.asarray(thetas).transpose(), ['1','2','4','8'], './Q3/{0}.svg'.format(columns[0]), x='Number of clients', y=columns[0])
    plotSeries(np.asarray(pps).transpose(), ['1','2','4','8'], './Q3/{0}.svg'.format(columns[1]), x='Number of clients', y=columns[1])
    plotSeries(np.asarray(cp).transpose(), ['1','2','4','8'], './Q3/{0}.svg'.format(columns[2]), x='Number of clients', y=columns[2])
    plotSeries(np.asarray(d).transpose(), ['1','2','4','8'], './Q3/{0}.svg'.format(columns[3]), x='Number of clients', y=columns[3])

plotQ2()
