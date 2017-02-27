import requests
import AdvancedHTMLParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import setp
import copy
from sklearn.linear_model import LinearRegression

parser = AdvancedHTMLParser.AdvancedHTMLParser()

param={'sciper': 223720, 'clients': 1, 'apoints': 1, 'servers': 1}
columns=['Successful download requests per second', 'Packets per second', 'Collision probability', 'Delay']


def sameReq(nbReq, param): 
    results = []
    for i in range (nbReq):
        page = requests.post("http://tcpip.epfl.ch/perfeval-lab1/output.php", data=param)
        parser.parseStr(page.text)
        t = parser.getElementsByTagName('td')
        t = [x.innerHTML for x in t]
        results.append( [float(t[11]), float(t[13]), float(t[15]), float(t[17])] )
    return results

def deffReq(nbReq, param, field, step):
    results = []
    for i in range (nbReq):
        param[field] = i*step + 1
        page = requests.post("http://tcpip.epfl.ch/perfeval-lab1/output.php", data=param)
        parser.parseStr(page.text)
        t = parser.getElementsByTagName('td')
        t = [x.innerHTML for x in t]
        results.append( [float(t[11]), float(t[13]), float(t[15]), float(t[17])] )
    return results

def plotBox(data, labels, path, title='Test'):
    df = pd.DataFrame(data, columns=[labels])
    ax = df.plot.box(colormap='jet',title=title)
    ax.grid()
    plt.savefig(path, format='svg', dpi=800)

def plotSeries(data, labels, path, x, y, title='Test'):
    df = pd.DataFrame(data, columns=labels)
    ax = df.plot(lw=1,colormap='jet',marker='.',markersize=6,title=title, x=np.arange(1, 1000, 10))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid()
    plt.savefig(path, format='svg', dpi=800)

def plotQ1():
    t = sameReq(1000, param)
    t = zip(*t)
    t = zip(t, columns)
    for x in t:
        plotBox(list(x[0]), [x[1]], './Q1/{0}.svg'.format(x[1]))

def plotQ2():
    t = deffReq(100, param, 'clients', 10)
    t = zip(*t)
    t = zip(t, columns)
    for x in t:
        plotSeries(list(x[0]), [x[1]], './Q2/{0}.svg'.format(x[1]), x='Number of clients', y=x[1])

def plotQ3():
    params = [copy.deepcopy(param),
    copy.deepcopy(param),
    copy.deepcopy(param),
    copy.deepcopy(param)]

    params[1]['apoints'] = 2 * params[0]['apoints']
    params[2]['apoints'] = 2 * params[1]['apoints']
    params[3]['apoints'] = 2 * params[2]['apoints']

    thetas=[]; pps=[]; cp=[]; d=[]
    print('toz')
    for p in params:
        t = deffReq(100, p, 'clients', 10)
        print('r')
        thetas.append([x[0] for x in t])
        pps.append([x[1] for x in t])
        cp.append([x[2] for x in t])
        d.append([x[3] for x in t])
    plotSeries(np.asarray(thetas).transpose(), ['1','2','4','8'], './Q3/{0}.svg'.format(columns[0]), x='Number of clients', y=columns[0])
    plotSeries(np.asarray(pps).transpose(), ['1','2','4','8'], './Q3/{0}.svg'.format(columns[1]), x='Number of clients', y=columns[1])
    plotSeries(np.asarray(cp).transpose(), ['1','2','4','8'], './Q3/{0}.svg'.format(columns[2]), x='Number of clients', y=columns[2])
    plotSeries(np.asarray(d).transpose(), ['1','2','4','8'], './Q3/{0}.svg'.format(columns[3]), x='Number of clients', y=columns[3])


def test():
    t = deffReq(100, param, 'clients', 10)
    theta = [x[0] for x in t]
    pps = [x[1] for x in t]
    ratio = [y[0] / y[1] if y[1] != 0 else 0 for y in zip(theta,pps)]
    X = np.asarray(range(len(ratio))).reshape(len(ratio), 1)
    Y = np.asarray(ratio).reshape(len(ratio), 1)
    reg = LinearRegression() 
    reg.fit(X, Y)

    '''plt.ylim(0, 0.3)
    plt.scatter(X, Y,  color='blue', s=2)
    plt.plot(X, reg.predict(X), color='red', linewidth=2)

    plt.savefig('./test/ratio.svg', format='svg', dpi=1200)'''

    df1 = pd.DataFrame(Y)
    df2 = pd.DataFrame(reg.predict(X))

    ax = df1.plot(lw=1,colormap='jet',marker='.',markersize=6, x=np.arange(1, 1000, 10), title='Ratio of successful download per number of packets sent')
    df2.plot(lw=1, x=np.arange(1, 1000, 10), ax=ax, color='red')
    ax.grid()

    plt.savefig('./test/ratio.svg', format='svg', dpi=800)

    


#plotQ1()
#plotQ2()
#plotQ3()
test()