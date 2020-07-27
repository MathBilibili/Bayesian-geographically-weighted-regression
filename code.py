# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:07:18 2020

@author: Yang Liu
"""

import numpy as np
import os
import pandas as pd
from scipy.special import loggamma
import scipy.stats
from datetime import datetime
from multiprocessing import Pool

is_para = True

#os.chdir("D:\\360download\\nus_statistics\\Cam_biostat\\Yangs_report\\200701\\code")

# set multiple cores
#pool = ThreadPool(4)

#geographical kernel bandwidth
h = 1

#Geographically weighted kernel
def G_kernel(d,h):
    return(np.exp(-d**2/h**2))
    
#euclidean distance
def eucliDis(A,B):
    A = np.array(A)
    B = np.array(B)
    return np.sqrt(sum((A-B)**2))

#log-likelihood negative binomial 
def negBion(outcome,offset,covariate,phi,theta):
    mean = np.exp(np.log(offset)+sum(np.array(covariate)*np.array(phi)))
    result = loggamma(outcome+1/theta)-loggamma(1/theta)-loggamma(outcome+1)-(1/theta)*np.log(1+theta*mean)+outcome*np.log(theta*mean/(1+theta*mean))
    return result

#log-prior phi (uniform)
def prior_phi(phi):
    if max(abs(np.array(phi)))>1000:
        return np.log(0)
    else:
        return np.log(1/2000)*len(phi)
    
#log-prior theta (uniform 0,1000)
def prior_theta(theta):
    if theta>1000 or theta<=0:
        return np.log(0)
    else:
        return np.log(1/1000)
    
#proposal phi
def r_phi(phi):
    phi = np.array(phi)
    phi_n = np.random.normal(loc=phi,scale=1,size=len(phi))
    return phi_n

def d_phi(phi_n,phi):
    return sum(np.log(scipy.stats.norm(phi, 1).pdf(phi_n)))

#proposal theta
def r_theta(theta):
    lower, upper, sd = 0, 1000, 0.1
    X = scipy.stats.truncnorm(
          (lower-theta)/sd,(upper-theta)/sd,loc=theta,scale=sd)
    return float(X.rvs(size=1))

def d_theta(theta_n,theta):
    theta_n = np.array(theta_n)
    theta = np.array(theta)
    lower, upper, sd = 0, 1000, 0.1
    X = scipy.stats.truncnorm(
          (lower-theta)/sd,(upper-theta)/sd,loc=theta,scale=sd)
    return sum(np.log(X.pdf(theta_n)))


#read data
data = pd.read_csv('simulateDate.csv',encoding='utf-8',header=0)

location = data[['x','y']].drop_duplicates(subset=['x','y'])
num_location = location.shape[0]
location['index']=range(num_location)


def weight_like(data_slice,loc_int,phi,theta,h):
    loc1=data_slice[0:2]
    theta_ind = int(location.loc[(location['x']==loc1[0]) & (location['y']==loc1[1])]['index'])
    dis = eucliDis(loc1,loc_int)
    kern = G_kernel(dis,h)
    return(kern*negBion(data_slice[2],data_slice[3],data_slice[4:],phi,theta[theta_ind]))

def joint_like(data,loc_int,phi,theta,h):
    slice_like = lambda x: weight_like(x,loc_int,phi,theta,h)
    #result = sum( data.apply(slice_like,axis=1) )
    result = sum(map(slice_like,data.values))
    result += (prior_phi(phi) + sum(list(map(prior_theta, theta))))
    return(result)
    
#init=[[1,1],[1]*num_location]


def GWR_MCMC(data,loc_int,h,init,num_iter,thin,burn_in):
    phi_old = init[0]
    theta_old = init[1]
    sto_phi = np.zeros([(num_iter-burn_in)//thin,len(phi_old)])
    sto_theta = np.zeros((num_iter-burn_in)//thin)
    theta_focus = int(location.loc[(location['x']==loc_int[0]) & (location['y']==loc_int[1])]['index'])
    
    for i in range(num_iter):
        if( (i<=(burn_in-1)) | ((i+1) % thin !=0)):
            phi_new = r_phi(phi_old)
            theta_new = list(map(r_theta,theta_old))
            rate = joint_like(data,loc_int,phi_new,theta_new,h) + d_phi(phi_old,phi_new) + d_theta(theta_old,theta_new) - joint_like(data,loc_int,phi_old,theta_old,h) - d_phi(phi_new,phi_old) - d_theta(theta_new,theta_old)
            alfa = min(1,np.exp(rate))
            runif = np.random.uniform(0,1,1)[0]
            phi_old = phi_new if runif < alfa else phi_old
            theta_old = theta_new if runif <alfa else theta_old
        else:
            phi_new = r_phi(phi_old)
            theta_new = list(map(r_theta,theta_old))
            rate = joint_like(data,loc_int,phi_new,theta_new,h) + d_phi(phi_old,phi_new) + d_theta(theta_old,theta_new) - joint_like(data,loc_int,phi_old,theta_old,h) - d_phi(phi_new,phi_old) - d_theta(theta_new,theta_old)
            alfa = min(1,np.exp(rate))
            runif = np.random.uniform(0,1,1)[0]
            phi_old = phi_new if runif < alfa else phi_old
            theta_old = theta_new if runif <alfa else theta_old
            sto_phi[((i+1-burn_in)//thin) - 1] = phi_old
            sto_theta[((i+1-burn_in)//thin) - 1] = theta_old[theta_focus]            
            print([i,phi_old])
    result = {'phi':sto_phi,'theta':sto_theta}
    return(result)
    

    
def GWR_update(model_info):
    phi_old = model_info[0]
    theta_old = model_info[1]
    loc_int = model_info[2]
    joint_old = model_info[3]
    theta_focus = int(location.loc[(location['x']==loc_int[0]) & (location['y']==loc_int[1])]['index'])
    phi_new = r_phi(phi_old)
    theta_new = list(map(r_theta,theta_old))
    joint_new = joint_like(data,loc_int,phi_new,theta_new,h)
    rate = joint_new + d_phi(phi_old,phi_new) + d_theta(theta_old,theta_new) - joint_old - d_phi(phi_new,phi_old) - d_theta(theta_new,theta_old)
    alfa = min(1,np.exp(rate))
    runif = np.random.uniform(0,1,1)[0]
    phi_old = phi_new if runif < alfa else phi_old
    theta_old = theta_new if runif <alfa else theta_old
    joint_old = joint_new if runif <alfa else joint_old
    sto_theta = theta_old[theta_focus]
    return([list(phi_old),theta_old,loc_int,joint_old,sto_theta])
    
init = [[[1,1,1],[1]*num_location,list(x),joint_like(data,x,[1,1,1],[1]*num_location,h)] for x in location[['x','y']].values] 

# redefine weighted likelihood function for a single theta
def weight_like_s(data_slice,loc_int,phi,theta,h):
    loc1=data_slice[0:2]
    dis = eucliDis(loc1,loc_int)
    kern = G_kernel(dis,h)
    return(kern*negBion(data_slice[2],data_slice[3],data_slice[4:],phi,theta))

def GWR_MCMC_multloc(init,num_iter,thin,burn_in):
    sto_phi = np.zeros([(num_iter-burn_in)//thin,num_location,len(init[0][0])])
    sto_theta = np.zeros([(num_iter-burn_in)//thin,num_location,1])
    iter_param = init   
    MC_log = np.zeros(num_location)
    log_MC = np.zeros(num_location)
    WAIC_one = np.zeros(num_location)
    WAIC_two = np.zeros(num_location)
    for i in range(num_iter):
        if( (i<=(burn_in-1)) | ((i+1) % thin !=0)):
            if(is_para):
                iter_param = list(pool.map(GWR_update,iter_param))
            else:
                iter_param = list(map(GWR_update,iter_param))
        else:
            if(is_para):
                iter_param = list(pool.map(GWR_update,iter_param))
            else:
                iter_param = list(map(GWR_update,iter_param))
            for j in range(num_location):
                sub_log_lik = 0
                loc_foc = list(location.loc[location['index']==j][['x','y']].values[0])
                subdata=data.loc[(data['x']==loc_foc[0]) & (data['y']==loc_foc[1])]
                sto_phi[((i+1-burn_in)//thin) - 1][j] = iter_param[j][0]
                sto_theta[((i+1-burn_in)//thin) - 1][j] = iter_param[j][4] 
                slice_like = lambda x: weight_like_s(x,loc_foc,iter_param[j][0],iter_param[j][4],h)
                sub_log_lik = sum( subdata.apply(slice_like,axis=1) )
                MC_log[j] += sub_log_lik
                log_MC[j] += np.exp(sub_log_lik)
                WAIC_one[j] = MC_log[j]/((i+1-burn_in)//thin)
                WAIC_two[j] = np.log(log_MC[j]/((i+1-burn_in)//thin))
            WAIC = np.mean(2*WAIC_two - WAIC_one)
            print('{0}% complete. The WAIC is: {site}'.format((i+1)*100/num_iter, site=WAIC), flush=True)
    result = {'phi':sto_phi,'theta':sto_theta,'WAIC':WAIC}
    return(result)
    
    
time_one = datetime.now()
if __name__ == '__main__':
    pool = Pool(processes=10)
    re=GWR_MCMC_multloc(init,10,1,0)
time_two = datetime.now()

print(time_two-time_one)

est_phi=sum(re['phi'])/re['phi'].shape[0]
est_theta=sum(re['theta'])/re['theta'].shape[0]

phi_trace = np.zeros(shape=[re['phi'].shape[0],3])
for k in range(re['phi'].shape[0]):
    phi_trace[k] = re['phi'][k][re['phi'][0].shape[0]//2]
np.savetxt('phi_trace.csv',phi_trace,delimiter=',')

print(est_phi)
np.savetxt("est_phi.csv", est_phi, delimiter=",")
print(est_theta)
np.savetxt("est_theta.csv", est_theta, delimiter=",")
print(re['WAIC'])
