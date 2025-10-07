import numpy as np



class domain:
    def __init__(self,dim,X_train):
        factor=10**(-4)
        self.X_train=X_train
        self.dim=dim
        
        self.upper=[]
        self.lower=[]
        for dd in range(dim):
            self.upper.append(np.quantile(X_train[:,dd],1-factor))
            self.lower.append( np.quantile(X_train[:,dd],factor))
        self.upper=np.array(self.upper)
        self.lower=np.array(self.lower) 
        self.upper= self.upper+ 0.001* (self.upper-self.lower)
        self.lower= self.lower- 0.001* (self.upper-self.lower)
        #print(self.upper)
        #print(self.lower)
        self.difference =self.upper-self.lower
        self.density_factor=np.prod(self.difference)
        
        
    def transform_to_0_1(self, x_vec ):
        
        
        return [( x_vec[dd]-self.lower [dd])/(self.difference[dd]) for dd in range(self.dim)]


    def compute_data(self,XX):
        X_transform= (XX-self.lower)/self.difference
        return  X_transform 
    
    def transform_density_val(self, val):
        return val/self.density_factor
