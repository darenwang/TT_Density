import numpy as np 

from scipy import stats


class KL:
    def __init__(self):
        pass
    def compute( self,y_predict, y_true):
        eps=1e-8

        return -1*np.mean(  np.log( np.maximum(np.maximum( y_predict,0) /y_true , eps    )))
    
    
    
class kernel_density():
    def __init__(self,data):
        self.kernel = stats.gaussian_kde(np.array(data).transpose())
    def compute(self, X_new):
        return self.kernel(np.array(X_new).transpose())
