# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:31:04 2018

@author: johan
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
x=np.arange(1,100.)/50.

def weib(x,beta,eta,gamma):
    return (beta/eta)*(x-gamma/eta)**(beta-1)*np.exp(-((x-gamma)/eta)**beta)


loc, scale = 10, 1
s = np.random.logistic(loc, scale, 10000)
count, bins, ignored = plt.hist(s, bins=50)
plt.show
#%%
NSIM=1000
GAM=0.9999
#TOV=209.767
ML=20
OGAP=0.2
PV=250 # present value i.e. full investment cost

PV_AGP=np.zeros(NSIM)
for t in range(1, 22):
    mu, sigma = 0.233, 0.068
    OG=np.random.normal(mu,sigma,NSIM)
    mu, sigma = 7153, 1000
    SP=np.random.normal(mu,sigma,NSIM)
    mu, sigma = 2.7, .1
    OD=np.random.normal(mu,sigma,NSIM)
    mu, sigma = 77.69e6, 2e6
    RA=np.random.normal(mu,sigma,NSIM)
    mu, sigma = 140e6, 5e6
    OA=np.random.normal(mu,sigma,NSIM)
    mu, sigma = 90, 2
    MiR=np.random.normal(mu,sigma,NSIM)
    mu, sigma = 90, 2
    PR=np.random.normal(mu,sigma,NSIM)
    mu, sigma = 93, 2
    MeR=np.random.normal(mu,sigma,NSIM)
    mu, sigma = 3.25, .25
    SC=np.random.normal(mu,sigma,NSIM)
    mu, sigma = 3.25, .25
    EC=np.random.normal(mu,sigma,NSIM)
    mu, sigma = 4.5, .25
    PC=np.random.normal(mu,sigma,NSIM)
    mu, sigma = 100, 10
    MC=np.random.normal(mu,sigma,NSIM)
    beta, shift = 1.3063, 3.2067
    IR=shift+np.random.exponential(beta,NSIM)

    TOV=OA+RA
    
    PV_AGP +=((MiR*PR*MeR*TOV*OD*OG*SP/(GAM*ML*1e6)-(TOV*OD*MeR*EC+OA*SC*100)/(ML*100)
    - TOV*OD*MiR*PC/(ML*100) - PR*TOV*OD*MiR*OG*MC/(OGAP*ML*100*100))/1e9)*(1+IR/100)**-t

NPV = PV_AGP-PV

count, bins, ignored = plt.hist(NPV,30,normed=True)

#count, bins, ignored = plt.hist(s,30,normed=True)
#plt.plot(bins,1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins-mu)**2/(2*sigma**2)),
#         linewidth=2,color='r')
#plt.show