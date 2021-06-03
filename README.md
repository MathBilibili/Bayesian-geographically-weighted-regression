# Bayesian geographically weighted regression
This is the Python code to conduct Bayesian geographically weighted regression proposed in the paper "Generalized Geographically Weighted Regression Model within a Modularized Bayesian Framework". It includes code to replicate the results presented in section "Simulation" and "Application to real dataset".

# Authors
Yang Liu and Robert Goudie

MRC Biostatistics Unit, University of Cambridge

## Reference

The paper can be accessed via:

Liu, Yang and Robert J.B. Goudie. “Generalized Geographically Weighted Regression Model within a Modularized Bayesian Framework.” (2021). [	arXiv:2106.00996][arXiv] 	

# Abstract of the proposed model
Geographically weighted regression (GWR) models handle geographical dependence through a spatially varying coefficient model and have been widely used in applied science, but its Bayesian extension is unclear because it involves a weighted log-likelihood which does not imply a probability distribution on data. We present a Bayesian GWR model and show that its essence is dealing with partial misspecification of the model. Current modularized Bayesian inference methods accommodate partial misspecification from single component of the model. We extend these methods to handle partial misspecification in more than one component of the model, as required for our Bayesian GWR model. Information from the various spatial locations is manipulated via a geographically weighted kernel and the optimal manipulation is chosen according to a Kullback–Leibler (KL) divergence. We justify the model via an information risk minimization approach and show the consistency of the proposed estimator in terms of a geographically weighted KL divergence.
