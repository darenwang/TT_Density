import numpy as np 

from scipy import stats


class KL:
    def compute( y_predict, y_true):
        eps=1e-8
        
        diffs = np.log(np.maximum(y_true,    eps)) - np.log(np.maximum(y_predict, eps) )
        # average over all samples
        return np.mean(diffs)
    
    
    
class kernel_density():
    def __init__(self,data):
        self.kernel = stats.gaussian_kde(np.array(data).transpose())
    def compute(self, X_new):
        return self.kernel(np.array(X_new).transpose())


class KL_divergence():
    def __init__(self):
        pass

    def compute(self, y_true, y_test):
        return sum(y_true* (np.log(y_true) - np.log(y_test)) )/len(y_true)

