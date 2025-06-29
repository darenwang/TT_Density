import numpy as np
from scipy.stats import multivariate_normal

from scipy.stats import bernoulli




class gaussian_mixture():
    def __init__(self,dim, mean,cov):
        self.dim=dim
        self.mean_1=[mean[0] for __ in range(dim)]
        self.mean_2=[mean[1] for __ in range(dim)]
        self.cov_1 = np.diag( np.ones(dim)*cov[0])
        self.cov_2 = np.diag( np.ones(dim)*cov[1])
        self.prob=0.5
    
    def value(self,x_input):

        return self.prob*multivariate_normal.pdf(x_input, self.mean_1, self.cov_1)+ (1-self.prob)*multivariate_normal.pdf(x_input, self.mean_2, self.cov_2)
        

    def generate(self,N):
        

        rr= np.tensordot([bernoulli.rvs(self.prob, size=N)] , [np.ones(self.dim)]  ,axes=[[0],[0]]  )
        #data1= np.multiply(rr,np.random.uniform(0,1, N*self.dim).reshape((N, self.dim)))
        data1 = np.multiply( rr,np.random.multivariate_normal(self.mean_1,self.cov_1,N ) )

        data2 = np.multiply( -1*rr+1,np.random.multivariate_normal(self.mean_2,self.cov_2,N ))
        
        return data1+data2
   
