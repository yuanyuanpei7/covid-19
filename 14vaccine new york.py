import pandas as pd
import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#import numpy as np
import datetime as dt
from datetime import datetime
plt.rcParams['font.family'] = ['Microsoft JhengHei']
plt.rcParams.update({'font.size': 15})

def SEIR_model_immune(Y,t,beta,q,a,sigma,epsilon,gamma):
    S,E,I,R = Y

    if t>=100:

       
        a=a+(t-100)*0.3/101

        sigma=0.4
        
    dS = - beta * S * (1-a) * I / N - beta * S * (1-a) * q * E / N - beta * a * S * (1-sigma) * I / N - beta * a * S * (1-sigma) * q * E / N - sigma * a * S
    dE = beta * S * (1-a) * I / N + beta * S * (1-a) * q * E / N + beta * a * S * (1-sigma) * I / N + beta * a * S * (1-sigma) * q * E / N  - epsilon * E
    dI = epsilon * E - gamma * I
    dR = sigma * a * S + gamma * I
    return dS,dE,dI,dR

def SEIR_model(Y,t,beta,q,a,sigma,epsilon,gamma):
    S,E,I,R = Y
    
    dS = - beta * S * (1-a) * I / N - beta * S * (1-a) * q * E / N - beta * a * S * (1-sigma) * I / N - beta * a * S * (1-sigma) * q * E / N - sigma * a * S
    dE = beta * S * (1-a) * I / N + beta * S * (1-a) * q * E / N + beta * a * S * (1-sigma) * I / N + beta * a * S * (1-sigma) * q * E / N  - epsilon * E
    dI = epsilon * E - gamma * I
    dR = sigma * a * S + gamma * I
    return dS,dE,dI,dR

def fit_odeint(x,beta,q,a,sigma,epsilon,gamma):
    return integrate.odeint(SEIR_model, N0, x, args=(beta,q,a,sigma,epsilon,gamma))[:,2]

def fit_odeint_immnune(x,beta,q,a,sigma,epsilon,gamma):
    return integrate.odeint(SEIR_model_immune, N0, x, args=(beta,q,a,sigma,epsilon,gamma))[:,2]

co=pd.read_csv('./14vaccine new york.csv',encoding='gbk',header=0)
for index, row in co.iterrows():
    nation=row['Province_State']+', '+row['Country_Region']
    row=row.drop(['Province_State','Admin2','Country_Region','UID','FIPS','iso2','iso3','code3'])

    xlist=[]
    ylist=[]
    for index,val in row.items():
        if val>0:
            date = datetime.strptime(index,'%Y年%m月%d日')
            xlist.append(date)
            ylist.append(val)
    population = 19440469
    N=population
    x=np.array(xlist)
    y=np.array(ylist)
    if len(y)==0:
        continue
T = np.arange(len(y))
I0 = y[0]
E0 = I0/3
R0 =331736
N0 = population - E0 - I0 - R0, E0, I0, R0


anealingend=100

popt, pcov = optimize.curve_fit(fit_odeint, T[0:anealingend], y[0:anealingend], bounds=([0.01,0.01,0.01,0.4,0.15,0.06],[0.9,0.8,1,1,0.3,0.25]),maxfev=20000000)

poptall, pcov = optimize.curve_fit(fit_odeint, T, y, bounds=([0.01,0.01,0.01,0.01,0.15,0.06],[0.9,0.8,1,1,0.3,0.25]),maxfev=20000000)

fittedActural = fit_odeint(np.array(list(range(1,len(y)+75))), *popt)

fitted_immune = fit_odeint_immnune(np.array(list(range(1,len(y)+75))), *popt)
fitted_all = fit_odeint(np.array(list(range(1,len(y)+75))), *poptall)

fig, ax = plt.subplots()

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=(1, 15)))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y %b %d'))
plt.title('Infections / day in '+nation)

plt.plot(x,y,'r-',label='Infections')


plt.plot(x,fittedActural[0:len(y)],'b-',label='Prediction')

plt.plot(x[anealingend:len(y)],fitted_immune[anealingend:len(y)],'g-',label='Vaccines',linewidth=3)

plt.legend()
plt.savefig(nation+'.tif')
APlist=[]
for step in [30, 45, 60, 75,90]:

    vaccine_No=sum(fittedActural[0:anealingend+step])

    vaccine_Yes_predicted=sum(fitted_immune[0:anealingend+step])
    vaccine_yes_Actural=sum(y[0:anealingend+step])

    Actural_Decrease=(vaccine_yes_Actural-vaccine_No)/vaccine_No

    Predict_Decrease=(vaccine_Yes_predicted-vaccine_No)/vaccine_No

    APlist.append([step/30,Actural_Decrease,Predict_Decrease])

APlist=np.array(APlist)
    

