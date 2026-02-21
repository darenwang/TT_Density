import numpy as np



################################
class polynomial:
    def __init__(self,n, alpha) -> None:
        from numpy.polynomial.legendre import leg2poly
        self.n=n
        coef_mat = np.zeros((n, n), dtype=float)
        for j in range(1,n+1):
            c=[0 for __ in range(j)]
            c[-1]=1
            coef_mat[j-1, :len(c)] =leg2poly(c)
        #print(coef_mat)
        normalized_factor= np.sqrt(2) *np.sqrt((2*np.arange(n)+1)/2)
        
        alpha_vec=alpha*np.ones(n)
        alpha_vec[0]=1

        
        
        factors = alpha_vec* normalized_factor
        self.coef_mat_alpha = coef_mat*factors[:, None]
        
    
    def multivariate_all_basis_alpha(self, x):
        
        
        new_x=2*x -1
        
        #powers_x is np array of size d by n
        powers_x = np.vander(new_x,self.n, increasing=True)
        return powers_x @ self.coef_mat_alpha.T
    
    def scalar_all_basis_alpha(self, x):
        new_x = 2.0 * x - 1.0
        powers_x = new_x ** np.arange(self.n)                 # shape (n,)
        return powers_x @ self.coef_mat_alpha.T 


#n=5
#polynomial(n,1).multivariate_all_basis_alpha( np.array([0.5,0.3]))
########## 


class generate_basis_mat:
    def __init__(self, n, dim,alpha,data):
        self.n = n
        self.polynomial = polynomial(n,alpha)
        self.dim=dim
        self.data=data
        
    #single input of x of size dim
    def compute(self ):
        new_data= self.data.reshape(-1)
        new_data=new_data*2-1
        power_data= np.vander(new_data,self.n, increasing=True)
        basis_mat_flat= power_data @ self.polynomial.coef_mat_alpha.T
        
        return basis_mat_flat.reshape(len(self.data), self.dim, self.n)
    

 