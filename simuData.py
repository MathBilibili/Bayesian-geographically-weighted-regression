# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:28:52 2020

@author: Yang Liu
"""

import numpy as np
import os
import pandas as pd
from scipy.special import gamma
import scipy.stats

#os.chdir("D:\\360download\\nus_statistics\\Cam_biostat\\Yangs_report\\200701\\code")

def rv_binom(offset,covariate,phi,theta):
    r=1/theta
    p= r/(np.exp(np.log(offset)+sum(np.array(covariate)*np.array(phi)))+r)
    return(scipy.stats.nbinom(r,p).rvs(1))


re = pd.DataFrame({'x':[],'y':[],'outcome':[],'offset':[],'x1':[],'x2':[],'x3':[]})
true_param = pd.DataFrame({'x':[],'y':[],'phi1':[],'phi2':[],'theta':[]})
offset=1

for i in range(10):
    for j in range(10):
        x=i+1
        y=j+1
        phi=[3,0.02*np.sqrt((x)**2+(y)**2),0.05*(np.sin(x/2)+np.cos(y/2)+4)]
        theta = float(np.random.uniform(0.5,0.6,1))
        for n in range(20):
            covariate=[1,float(np.random.uniform(0,2,1)),float(np.random.uniform(2,7,1))]
            outcome = int(rv_binom(offset,covariate,phi,theta))
            re = re.append({'x':x,'y':y,'outcome':outcome,'offset':offset,'x1':covariate[0],'x2':covariate[1],'x3':covariate[2]},ignore_index=True)
            true_param = true_param.append({'x':x,'y':y,'phi1':phi[1],'phi2':phi[2],'theta':theta},ignore_index=True)
            print([i,j])

re.to_csv('simulateDate.csv')
true_param.to_csv('true_parameter.csv')
