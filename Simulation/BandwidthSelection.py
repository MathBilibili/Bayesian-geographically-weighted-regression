# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 14:08:10 2022

@author: Yang Liu
"""


import numpy as np
import os
import pandas as pd
from scipy.special import loggamma
import scipy.stats
from datetime import datetime
from multiprocessing import Pool

file_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
#print([task_id,type(task_id)],flush=True)

task_id = (file_id-1)//5
#task_id = (file_id-1)

rep_id = file_id%5
is_para = True    #Using parallel computing？
num_core = 30     #number of cores
fitting_ratio = 0.5    #the proportion of samples used for fitting at the location of interest (i.e., splitting samples at location of interest into fitting set and testing set).
is_eucliDis = True      #Using Euclidian distance?
is_block = False        #block sampling?


# set multiple cores
#pool = ThreadPool(4)

#geographical kernel bandwidth
h = [0.0001,2,4,6,8,10,20,40,80,1000][task_id]

print('h is {h}'.format(h=h), flush=True)

#Geographically weighted kernel (exponential kernel)
def G_kernel(d,h):
    return(np.exp(-d**2/h**2))
    
#euclidean distance
def eucliDis(A,B):
    A = np.array(A)
    B = np.array(B)
    return np.sqrt(sum((A-B)**2))

#spherical distance (measured in KM)
def Haversine(A,B):
    """
    This uses the ‘haversine’ formula to calculate the great-circle distance between two points – that is, 
    the shortest distance over the earth’s surface – giving an ‘as-the-crow-flies’ distance between the points 
    (ignoring any hills they fly over, of course!).
    Haversine
    formula:    a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    c = 2 ⋅ atan2( √a, √(1−a) )
    d = R ⋅ c
    where   φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km);
    note that angles need to be in radians to pass to trig functions!
    """
    lat1,lon1,lat2,lon2 = A[0],A[1],B[0],B[1]
    
    R = 6378.0088
    lat1,lon1,lat2,lon2 = map(np.radians, [lat1,lon1,lat2,lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) **2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    d = R * c
    return round(d,4)

#log-likelihood of negative binomial distribution
def negBion(outcome,offset,covariate,phi,theta):
    mean = np.exp(np.log(offset)+sum(np.array(covariate)*np.array(phi)))
    result = loggamma(outcome+1/theta)-loggamma(1/theta)-loggamma(outcome+1)-(1/theta)*np.log(1+theta*mean)+outcome*np.log(theta*mean/(1+theta*mean))
    return result

#log-prior for phi (uniform) 
def prior_phi(phi):
    if max(abs(np.array(phi)))>1000:
        return np.log(0)
    else:
        return np.log(1/2000)*len(phi)
    
#log-prior for theta (uniform 0,1000)
def prior_theta(theta):
    if theta>1000 or theta<=0:
        return np.log(0)
    else:
        return np.log(1/1000)
    
#baseline proposal sd for phi (proportional to estimated correlation matrix of phi)
#pro_st = np.array([[ 0.19800107, -0.03503183, -0.0323151],
#       [-0.03503183,  0.03827607, -0.00078348],
#       [-0.0323151 , -0.00078348,  0.00733836]])

pro_st = np.array([[ 0.199212513, -0.00319569314, -0.037421612],
       [-0.00319569314, 0.0079491569, 0.000190719635],
       [-0.037421612, 0.000190719635, 0.00757105334]])
    
#two step adatpive proposal sd:
#aggressive proposal sd for phi to approximate true value before burn_in
pro_early = [np.dot(pro_st*2,pro_st*2),np.dot(pro_st,pro_st),np.dot(pro_st,pro_st),np.dot(pro_st,pro_st),np.dot(pro_st,pro_st),np.dot(pro_st,pro_st),np.dot(pro_st,pro_st),np.dot(pro_st,pro_st),np.dot(pro_st,pro_st),np.dot(pro_st,pro_st)]
#mild proposal sd for phi to achieve a good mixture after burn_in
pro_later = [np.dot(pro_st*2,pro_st*2),np.dot(pro_st/5,pro_st/5),np.dot(pro_st/5,pro_st/5),np.dot(pro_st/5,pro_st/5),np.dot(pro_st/5,pro_st/5),np.dot(pro_st/5,pro_st/5),np.dot(pro_st/10,pro_st/10),np.dot(pro_st/10,pro_st/10),np.dot(pro_st/10,pro_st/10),np.dot(pro_st/10,pro_st/10)]
 
#aggressive proposal sd for theta to approximate true value before burn_in 
pro_theta_early = [0.05,0.03,0.03,0.03,0.015,0.015,0.005,0.004,0.004,0.004]
#mild proposal sd for theta to achieve a good mixture after burn_in
pro_theta_later = [0.05,0.015,0.015,0.015,0.005,0.005,0.005,0.004,0.004,0.004]

#proposal sampling function for phi (multivariate normal)
def r_phi(phi):
    phi = np.array(phi)
    phi_n = scipy.stats.multivariate_normal(phi,pro_early[task_id]).rvs(1)
    return phi_n
#proposal density function for phi
def d_phi(phi_n,phi):
    return np.log(scipy.stats.multivariate_normal(phi,pro_early[task_id]).pdf(phi_n))

#proposal sampling function for phi (multivariate normal)
def r_phi_new(phi):
    phi = np.array(phi)
    phi_n = scipy.stats.multivariate_normal(phi,pro_later[task_id]).rvs(1)
    return phi_n
#proposal density function for phi
def d_phi_new(phi_n,phi):
    return np.log(scipy.stats.multivariate_normal(phi,pro_later[task_id]).pdf(phi_n))

#proposal sampling function for theta (truncated normal)
def r_theta(theta):
    sd = pro_theta_early[task_id]
    out = np.random.normal(theta, sd, len(theta))
    out[out<=0] = 0.001
    return out

#proposal density function for theta
def d_theta(theta_n,theta):
    sd =  pro_theta_early[task_id]
    return sum(scipy.stats.norm.pdf(theta_n, theta, sd))

#proposal sampling function for theta (truncated normal)
def r_theta_new(theta):
    sd = pro_theta_later[task_id]
    out = np.random.normal(theta, sd, len(theta))
    out[out<=0] = 0.001
    return out

#proposal density function for theta
def d_theta_new(theta_n,theta):
    sd =  pro_theta_later[task_id]
    return sum(scipy.stats.norm.pdf(theta_n, theta, sd))



#import data from file
data = pd.read_csv('simulateDate.csv',encoding='utf-8',header=0)
#extract coordinates of locations
location = data[['x','y']].drop_duplicates(subset=['x','y'])
#number of locations
num_location = location.shape[0]
#add main key to location table
location['index']=range(num_location)

#randomly select fitting subdata for each location of interest (cross validation)
index_sel = []

for k in range(num_location):
    loc_foc = location.values[k][0:2]
    index_loc = data[(data['x']==loc_foc[0]) & (data['y']==loc_foc[1])].index
    num_sel = int(len(index_loc)*fitting_ratio)
    index_sel.append(np.sort(np.random.choice(index_loc,size=num_sel,replace=False)))
    
#given a observation (data_slice) and a location of interest (loc_int), this function calculate the geographical weight
def kernel_weight(data_slice,loc_int,h):
    loc1=data_slice[0:2]
    if(is_eucliDis == True):
        dis = eucliDis(loc1,loc_int)
    else:
        dis = Haversine(loc1,loc_int)
    kern = G_kernel(dis,h)
    return(kern)

#geographical weighting kernel matrix
#In this part, we store all geographical weights in matrix (or list) "geo_weight" to avoid redundant calculation. The number of rows is equal to the number of location of interest. At each location of interest (e.g., each row), we calculate the geographical weight for each sample.
weight = []
theta_rep_num = np.zeros(shape=[num_location,num_location])
theta_slice_ind = np.zeros(shape=[num_location,(num_location+1)])

theta_slice_ind_full = [np.equal(data.iloc[:,:2].values, location[['x','y']].values[i]).all(axis=1).sum() for i in range(num_location)]
theta_slice_ind_drop = []
for i in range(num_location):
    theta_slice_ind_drop.append(np.equal(data.drop(index_sel[i]).iloc[:,:2].values, location[['x','y']].values[i]).all(axis=1).sum())
for i in range(num_location):
    theta_rep_num[i] = theta_slice_ind_full
    theta_rep_num[i][i] = theta_slice_ind_drop[i]

for i in range(num_location):
    loc_int_inner = location.values[i][0:2]
    slice_weight = lambda x: kernel_weight(x,loc_int_inner,h)
    weight.append(np.array([0]*i+list(map(slice_weight,data.drop_duplicates(subset=['x','y']).iloc[i:,:].values))))
    theta_slice_ind[i] = np.append(0,[sum(theta_rep_num[i][:(k+1)]) for k in range(len(theta_rep_num[i]))])


theta_slice_ind = theta_slice_ind.astype(int)        
weight = np.array(weight)        
weight = weight + weight.T - np.eye(num_location)
geo_weight = [np.repeat(weight[i],theta_rep_num[i].astype(int)) for i in range(num_location)]


#data index that is omitted due to very low weight:
minimum_threshold = 10**(-1)
drop_set = []
theta_num = np.zeros(num_location)
for j in range(num_location):
    theta_num[j] = np.equal(data.iloc[:,:2].values, location[['x','y']].values[j]).all(axis=1).sum()
for i in range(num_location):
    set_one = index_sel[i]
    full_geo_weight = np.repeat(weight[i],theta_num.astype(int)) 
    set_two = np.where(full_geo_weight<minimum_threshold)[0]
    drop_set.append(np.array(list(set(set_one).union(set(set_two)))))

geo_weight_cut = [np.delete(np.repeat(weight[i],theta_num.astype(int)),drop_set[i]) for i in range(num_location)]

print("geo_weight has been calculated")

# joint log-density of negative binomial likelihood and prior given a certain location of interest.   Vectorization for fast calculation
def joint_like(data,loc_ind,phi,theta,if_cut):
    if(if_cut == True):
        theta_expand = np.repeat(np.array(theta),theta_num[loc_ind].astype(int)) 
        theta_expand = np.delete(theta_expand,drop_set[loc_ind])
        outcome = np.array(data.drop(drop_set[loc_ind])['outcome'])
        offset = np.array(data.drop(drop_set[loc_ind])['offset'])
        covariate = np.array(data.drop(drop_set[loc_ind]).iloc[:, 4:])
        mean = np.exp(np.log(offset) + np.array(list(map(sum,covariate*np.array(phi)))))
        result = sum(geo_weight_cut[loc_ind]*(loggamma(outcome+1/theta_expand)-loggamma(1/theta_expand)-loggamma(outcome+1)-(1/theta_expand)*np.log(1+theta_expand*mean)+outcome*np.log(theta_expand*mean/(1+theta_expand*mean))))
        result += (prior_phi(phi) + sum(list(map(prior_theta, theta))))
        return(result)
    else:
        theta_expand = np.repeat(np.array(theta),theta_rep_num[loc_ind].astype(int))   
        outcome = np.array(data.drop(index_sel[loc_ind])['outcome'])
        offset = np.array(data.drop(index_sel[loc_ind])['offset'])
        covariate = np.array(data.drop(index_sel[loc_ind]).iloc[:, 4:])
        mean = np.exp(np.log(offset) + np.array(list(map(sum,covariate*np.array(phi)))))
        result = sum(geo_weight[loc_ind]*(loggamma(outcome+1/theta_expand)-loggamma(1/theta_expand)-loggamma(outcome+1)-(1/theta_expand)*np.log(1+theta_expand*mean)+outcome*np.log(theta_expand*mean/(1+theta_expand*mean))))
        result += (prior_phi(phi) + sum(list(map(prior_theta, theta))))
        return(result)



def theta_like(subdata,loc_ind,phi,theta):
    theta_expand = np.repeat(np.array(theta),theta_rep_num[loc_ind].astype(int))   
    outcome = np.array(subdata['outcome'])
    offset = np.array(subdata['offset'])
    covariate = np.array(subdata.iloc[:, 4:])
    mean = np.exp(np.log(offset) + np.array(list(map(sum,covariate*np.array(phi)))))
    theta_like_value = geo_weight[loc_ind]*(loggamma(outcome+1/theta_expand)-loggamma(1/theta_expand)-loggamma(outcome+1)-(1/theta_expand)*np.log(1+theta_expand*mean)+outcome*np.log(theta_expand*mean/(1+theta_expand*mean)))
    result = np.array([sum(theta_like_value[theta_slice_ind[loc_ind][k]:theta_slice_ind[loc_ind][k+1]]) for k in range(num_location)])
    result += np.array(list(map(prior_theta, theta)))
    return(result)
    
    

#old code (discarded)
#def weight_like(data_slice,loc_int,phi,theta,h):
#    loc1=data_slice[0:2]
#    theta_ind = int(location.loc[(location['x']==loc1[0]) & (location['y']==loc1[1])]['index'])
#    dis = eucliDis(loc1,loc_int)
#    kern = G_kernel(dis,h)
#    return(kern*negBion(data_slice[2],data_slice[3],data_slice[4:],phi,theta[theta_ind]))


#def joint_like(data,loc_int,phi,theta,h):
#    slice_like = lambda x: weight_like(x,loc_int,phi,theta,h)
#    #result = sum( data.apply(slice_like,axis=1) )
#    result = sum(map(slice_like,data.values))
#    result += (prior_phi(phi) + sum(list(map(prior_theta, theta))))
#    return(result)
  
#init=[[1,1],[1]*num_location]
    

# redefine weighted likelihood function for a single sample, this function is used to calculate the likelihood value of samples from testing set (cross validation).
def weight_like_s(data_slice,phi,theta):
    return(negBion(data_slice[2],data_slice[3],data_slice[4:],phi,theta))

#Given necessary model information "model_info" (i.e., list of value of phi, value of theta, coordinates of location of interest, joint density, value of theta at location of interest, number of accepted proposals),
#function "GWR_update" updates old "model_info" by one step metropolis hasting. The output is new "model_info".
#Note that, "GWR_update" only update one location. Therefore, it will be applied in parallel for all locations. See following function "GWR_MCMC_multloc"
if is_block:
    def GWR_update(model_info):
        phi_old = model_info[0]
        theta_old = model_info[1]
        loc_int = model_info[2]
        loc_ind = int(location.loc[(location['x']==loc_int[0]) & (location['y']==loc_int[1])]['index'])
        subdata = data.drop(index_sel[loc_ind])
        joint_old = model_info[3]
        accept_num = model_info[5]
        if_cut = model_info[6]
        theta_focus = int(location.loc[(location['x']==loc_int[0]) & (location['y']==loc_int[1])]['index'])
        phi_new = r_phi(phi_old)
        theta_new = r_theta(theta_old)
        joint_new_phi = joint_like(data,loc_ind,phi_new,theta_old,if_cut)
        rate_phi = joint_new_phi + d_phi(phi_old,phi_new) - joint_old - d_phi(phi_new,phi_old) 
        alfa_phi = min(1,np.exp(rate_phi))
        runif = np.random.uniform(0,1,1)[0]
        phi_old = phi_new if runif < alfa_phi else phi_old
        accept_num = (accept_num + 1) if runif <alfa_phi else accept_num
        joint_new_theta = theta_like(subdata,loc_ind,phi_old,theta_new)
        joint_old_theta = theta_like(subdata,loc_ind,phi_old,theta_old)
        rate_theta = joint_new_theta + d_theta(theta_old,theta_new) - joint_old_theta - d_theta(theta_new,theta_old)
        alfa_theta = np.minimum(np.ones_like(rate_theta),np.exp(rate_theta))
        runif = np.random.uniform(0,1,len(alfa_theta))
        theta_pro = [theta_new[q] if runif[q] < alfa_theta[q] else theta_old[q] for q in range(num_location)]
        theta_old = theta_pro
        sto_theta = theta_old[theta_focus]
        joint_old = joint_like(data,loc_ind,phi_old,theta_old,True)
        return([list(phi_old),theta_old,loc_int,joint_old,sto_theta,accept_num,if_cut,0])
else:
    def GWR_update(model_info):
        phi_old = model_info[0]
        theta_old = model_info[1]
        loc_int = model_info[2]
        loc_ind = model_info[7]
        joint_old = model_info[3]
        accept_num = model_info[5]
        if_cut = model_info[6]
        phi_new = r_phi(phi_old)
        theta_new = r_theta(theta_old)
        joint_new = joint_like(data,loc_ind,phi_new,theta_new,if_cut)
        rate = joint_new + d_phi(phi_old,phi_new) + d_theta(theta_old,theta_new) - joint_old - d_phi(phi_new,phi_old) - d_theta(theta_new,theta_old)
        alfa = min(1,np.exp(rate))
        runif = np.random.uniform(0,1,1)[0]
        update_sample = [phi_new,theta_new,joint_new,(accept_num + 1)] if runif < alfa else [phi_old,theta_old,joint_old,accept_num]        
        phi_old, theta_old, joint_old, accept_num = update_sample[0],update_sample[1],update_sample[2],update_sample[3]
        sto_theta = theta_old[loc_ind]
        return([list(phi_old),theta_old,loc_int,joint_old,sto_theta,accept_num,if_cut,model_info[7],0])
        
    def GWR_update_new(model_info):
        phi_old = model_info[0]
        theta_old = model_info[1]
        loc_int = model_info[2]
        loc_ind = model_info[7]
        joint_old = model_info[3]
        accept_num = model_info[5]
        if_cut = model_info[6]
        phi_new = r_phi_new(phi_old)
        theta_new = r_theta_new(theta_old)
        joint_new = joint_like(data,loc_ind,phi_new,theta_new,if_cut)
        rate = joint_new + d_phi_new(phi_old,phi_new) + d_theta_new(theta_old,theta_new) - joint_old - d_phi_new(phi_new,phi_old) - d_theta_new(theta_new,theta_old)
        alfa = min(1,np.exp(rate))
        runif = np.random.uniform(0,1,1)[0]     
        update_sample = [phi_new,theta_new,joint_new,(accept_num + 1)] if runif < alfa else [phi_old,theta_old,joint_old,accept_num]        
        phi_old, theta_old, joint_old, accept_num = update_sample[0],update_sample[1],update_sample[2],update_sample[3]
        sto_theta = theta_old[loc_ind]
        #testing subdata
        subdata=data.iloc[index_sel[loc_ind]]     #testing set at location j
        slice_like = lambda x: weight_like_s(x,phi_old,sto_theta)     #define the likelihood function by using posterior phi and theta at location j of iteration i
        sub_log_lik = sum( subdata.apply(slice_like,axis=1) )       #calculate the likelihood of testing set at location j by using posterior phi and theta at iteration i
        return([list(phi_old),theta_old,loc_int,joint_old,sto_theta,accept_num,if_cut,model_info[7],sub_log_lik])
    
    

#initial value "init" (list of initial phi, initial theta, coordinates of location of interest, initial joint density)    
#init_phi = [2,1,1]  
#init_param = pd.read_csv('init_param.csv',encoding='utf-8',header=0)
#init = [[[init_param.iloc[x]['phi0'],init_param.iloc[x]['phi1'],init_param.iloc[x]['phi2']],[init_param.iloc[x]['theta']]*num_location,list(location[['x','y']].values[x]),joint_like(data,x,[init_param.iloc[x]['phi0'],init_param.iloc[x]['phi1'],init_param.iloc[x]['phi2']],[init_param.iloc[x]['theta']]*num_location,True),init_param.iloc[x]['theta'],0,True,x,0] for x in range(len(location[['x','y']].values))] 
init_phi = [2.8421216,0.4794747,0.2232922]
init_theta = 0.5
init = [[init_phi,[init_theta]*num_location,list(location[['x','y']].values[x]),joint_like(data,x,init_phi,[init_theta]*num_location,True),init_theta,0,True,x,0] for x in range(len(location[['x','y']].values))]


#MCMC updates of all location simultaneously (mignt be in parallel).
def GWR_MCMC_multloc(init,num_iter,thin,burn_in):
    sto_phi = np.zeros([(num_iter-burn_in)//thin,num_location,len(init[0][0])])     #store posterior samples of phi
    sto_theta = np.zeros([(num_iter-burn_in)//thin,num_location,1])     #store posterior samples of theta
    iter_param = init   
    loglik_sum = np.zeros(num_location)     #store the log-likelihood of testing set for all locations
    elpd = np.zeros(num_location)       #store the estimated elpd for all locations
    for i in range(num_iter):
        if( (i<=(burn_in-1)) | ((i+1) % thin !=0)):
            if(is_para):
                iter_param = list(pool.map(GWR_update,iter_param))      #one step metropolis hasting update for all locations in parallel
            else:
                iter_param = list(map(GWR_update,iter_param))       #one step metropolis hasting update for all locations
            if((i+1)%100==0):
                accept_rate = np.mean(np.array([iter_param[s][5] for s in range(num_location)])/i)
                print('The average acceptance rate is: {rate}'.format(rate=accept_rate),flush=True)
                
            if((i+1) % thin == 0):
                print('{0}% complete.'.format((i+1)*100/num_iter), flush=True)
        else:
            if(i%20==0):
                for s in range(num_location):
                    iter_param[s][6] = False
                drop_like = np.array([iter_param[q][3] for q in range(num_location)])

            if(is_para):
                iter_param = list(pool.map(GWR_update_new,iter_param))      #one step metropolis hasting update for all locations in parallel
            else:
                iter_param = list(map(GWR_update_new,iter_param))       #one step metropolis hasting update for all locations

            accept_rate = np.mean(np.array([iter_param[s][5] for s in range(num_location)])/i)     #calculate the mean acceptance rate
            sto_phi[((i+1-burn_in)//thin) - 1] = np.array([iter_param[s][0] for s in range(num_location)])
            sto_theta[((i+1-burn_in)//thin) - 1] = np.array([[iter_param[s][4]] for s in range(num_location)])
            aa = np.array([loglik_sum[s] + iter_param[s][8] for s in range(num_location)])
            loglik_sum = aa
            elpd = np.array([loglik_sum[s]/((i+1-burn_in)//thin) for s in range(num_location)])
            elpd_mean = np.mean(elpd)       #mean of elpd across all locations
            
            if(i%20==0):
                print(sto_phi[((i+1-burn_in)//thin) -1], flush=True)
                print('Theta at 800 is: {theta}'.format(theta=sto_theta[((i+1-burn_in)//thin) - 1][800]),flush=True)
                for s in range(num_location):
                    iter_param[s][6] = True
                full_like = np.array([iter_param[q][3] for q in range(num_location)])
                print('The approximation rate is: {rate}'.format(rate=np.mean(drop_like/full_like)),flush=True)
            print('{0}% complete. The ELPD is: {site}. The average acceptance rate is: {rate}'.format((i+1)*100/num_iter, site=elpd_mean, rate=accept_rate), flush=True)
    result = {'phi':sto_phi,'theta':sto_theta,'ELPD':elpd_mean}
    return(result)
    
    
time_one = datetime.now()
if __name__ == '__main__':
    pool = Pool(processes=num_core)
    re=GWR_MCMC_multloc(init,4000,1,1000)
time_two = datetime.now()

print(time_two-time_one)        #time used for MCMC updates

est_phi=sum(re['phi'])/re['phi'].shape[0]       #posterior estimation of phi (posterior mean)
est_theta=sum(re['theta'])/re['theta'].shape[0]     ##posterior estimation of theta (posterior mean)

trace = np.zeros(shape=[re['phi'].shape[0],re['phi'].shape[2]+re['theta'].shape[2]])        #trace record of posterior samples (central location only)
for k in range(re['phi'].shape[0]):
    trace[k][0:re['phi'].shape[2]] = re['phi'][k][re['phi'][0].shape[0]//2]
    trace[k][re['phi'].shape[2]:] = re['theta'][k][re['theta'][0].shape[0]//2]
#np.savetxt('trace'+str(h)+'.csv',trace,delimiter=',')

print(est_phi)
#np.savetxt("est_phi"+str(h)+".csv", est_phi, delimiter=",")
#print(est_theta)
#np.savetxt("est_theta"+str(h)+".csv", est_theta, delimiter=",")
print([h,re['ELPD']])
