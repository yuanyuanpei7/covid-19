import pandas as pd
import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime as dt
from datetime import datetime
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams.update({'font.size': 16})

def SEIR_model(Y,t,beta,q,a,sigma,epsilon,gamma):
    S,E,I,R = Y
    dS = - beta * S * (1-a) * I / N - beta * S * (1-a) * q * E / N - beta * a * S * (1-sigma) * I / N - beta * a * S * (1-sigma) * q * E / N - sigma * a * S
    dE = beta * S * (1-a) * I / N + beta * S * (1-a) * q * E / N + beta * a * S * (1-sigma) * I / N + beta * a * S * (1-sigma) * q * E / N  - epsilon * E
    dI = epsilon * E - gamma * I
    dR = sigma * a * S + gamma * I
    return dS,dE,dI,dR

def fit_odeint(x,beta,q,a,sigma,epsilon,gamma):
    return integrate.odeint(SEIR_model, N0, x, args=(beta,q,a,sigma,epsilon,gamma))[:,2]
co=pd.read_csv('./12Iran.csv',encoding='gbk',header=0)
for index, row in co.iterrows():
    nation=row['Country/Region']
    row=row.drop(['Province_State','Admin2','Country/Region','UID','FIPS','iso2','iso3','code3'])

    xlist=[]
    ylist=[]
    for index,val in row.items():
        if val>=0:
            date = datetime.strptime(index,'%Y年%m月%d日')
            xlist.append(date)
            ylist.append(val)
    population = 83992953
    N=population
    x=np.array(xlist)
    y=np.array(ylist)
    if len(y)==0:
        continue
    T = np.arange(len(y))
    I0 = y[0]
    E0 = I0/3
    R0 =1304330
    N0 = population - E0 - I0 - R0, E0, I0, R0

popt, pcov = optimize.curve_fit(fit_odeint, T[1:(len(y)-30)], y[1:(len(y)-30)], bounds=([0.01,0.01,0.01,0.01,0.15,0.06],[0.9,0.8,1,1,0.3,0.25]),maxfev=20000000)
fitted = fit_odeint(np.array(list(range(1,len(y)+1))), *popt)
fig, ax = plt.subplots()

ax.xaxis.set_major_locator(mdates.MonthLocator())

ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=(1, 15)))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.title('Infections/day in ' + nation)

plt.plot(x, y, 'r-', label='Real infections', linewidth=1)

plt.plot(x, fitted[:len(y)], 'y-', label='Fitting', linewidth=3)
plt.plot(x[(len(y) - 30):len(y)], fitted[(len(y) - 30):len(y)], 'b-', label='Prediction', linewidth=3)
plt.legend()
plt.savefig(nation + '.svg')