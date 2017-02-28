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
        res.append(mean(10, 1001, paramBase, 'clients', 1))
    
    res = np.asarray(res)

    np.save('./Q3/data', res)


def plotQ3():

    res = np.load('./Q3/data.npy')
    print(np.shape(res))
    resMean = np.mean(res, 1)
    print(np.shape(resMean))

    r = resMean.tolist()
    ap1, ap2, ap4, ap8 = [r[0], r[1], r[2], r[3]]

    def mergeAp(dim):
        return [[ap1[dim], ap2[dim], ap4[dim], ap8[dim]] for x in ap1]

    theta = mergeAp(1)
    print(np.shape(theta))

plotQ3()