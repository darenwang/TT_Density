import numpy as np 

from scipy import stats


 
    
class kernel_density():
    def __init__(self,data):
        self.kernel = stats.gaussian_kde(np.array(data).transpose())
    def compute(self, X_new):
        return self.kernel(np.array(X_new).transpose())
