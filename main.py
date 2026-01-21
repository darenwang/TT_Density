
import numpy as np






N=    10** 5
dim= 50
N_test=10**4


#alpha=1


print( 'sample size=', N)
print( 'dim=', dim)

#####################
#####################
#####################generate data

"""
from gaussian_mixture import gaussian_mixture
###Gaussian 
means=[-1, 1]
standard_deviations=[1,1]
generator = gaussian_mixture(dim, means, standard_deviations )

"""



#####beta mixture
from beta_mixture import beta_mixture
alpha  =  [2*np.ones(dim),  3*np.ones(dim),1*np.ones(dim), ] 

beta  =  [2 *np.ones(dim),  2*np.ones(dim),1*np.ones(dim), ] 
generator=beta_mixture(dim,   alpha , beta , [0.4, 0.3  , 0.3  ])



#######mixture_sin_cos



X_train=generator.generate(N)
X_test=generator.generate(N_test)
y_true=generator.value(X_test)


##############
###############################  TT prediction via Nystrom 
from nystrom import TT_svd
ranks=[3 for _ in range(dim-1)]
n= 8
alpha=1/np.sqrt(dim)/n 
s=300


    
    


y_TT= TT_svd(  n, ranks, alpha, s, X_train).predict(X_test)

TT_error= (np.linalg.norm(y_TT  -y_true)/np.linalg.norm( y_true) ) **2
print( 'TT error', TT_error) 


#kl = np.mean(np.log(np.maximum(y_true, 1e-7)) - np.log(np.maximum(y_TT, eps)))
#print(kl)

"""
#####  Cross Validation for  TT to chooce tuning parameters
from nystrom import nystrom_cv

ranks1=[3 for _ in range(dim-1)]
n1= 10
alpha1=1/np.sqrt(dim)/n 
s1=500
ranks2=[3 for _ in range(dim-1)]
n2=8
alpha2=1/np.sqrt(dim*n) 
s2=300
set_parameter=[[ranks1, n1,alpha1,s1],[ranks2, n2,alpha2,s2] ]

TT_density= nystrom_cv(N,dim  ,X_train).compute(set_parameter)
 
y_TT_cv= TT_density.predict(X_test)
TT_error_cv=  ( np.linalg.norm(y_TT_cv -y_true)/np.linalg.norm( y_true)) **2
print( 'TT cv error ', TT_error_cv) 
"""
 
########## KDE  very slow when N is large 

from density_utility import  kernel_density
kde=kernel_density(X_train)

y_kde= kde.compute(X_test)


y_kde=np.array(y_kde)
kde_error= (np.linalg.norm(y_kde-y_true)/ np.linalg.norm(y_true))**2
print('kde error', kde_error)


######MAF (need GPU to boost computation speed)
#from denmarf import DensityEstimate

#de = DensityEstimate(device="cpu", use_cuda=False)

#aa= DensityEstimate().fit(X_train, bounded=False)  
#logp = aa.score_samples(X_test)
#y_NN = np.exp(logp)
#print( 'NN error',  (np.linalg.norm(y_NN  -y_true)/np.linalg.norm( y_true)) **2)
#####################################
