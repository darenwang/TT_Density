import numpy as np

class wavelet_1D:
    def __init__(self, n ,alpha):

        self.n = n 
        index = np.arange(n )
        #given index j, compute the level and translation k
        # level of j  = floor(log2(j)) for j >= 1, levels[0]=0
        
        levels= np.zeros(n , dtype=int)
        l = np.arange(1, n)
        levels[l] = np.floor(np.log2(index[l])).astype(int)
        #print(levels)
        self.l=l
        # translations k[j] = j - 2^level[j]
        self.levels=levels[l]
        self.ks = l - 2**self.levels 
        # normalization 2^(level/2)
        self.powers_square_root = 2**(self.levels / 2)*alpha
        self.powers= 2**self.levels
        
        
    def psi(self, y):
        #the base case
        #y is a vector of size N
        mask = (y  >= 0) & ( y  < 1)
        
        p = np.where(y < 0.5, 1.0, -1.0)
        p[~mask] = 0.0
        return p
    
    def basis(self, X):
        #X is N by 1
        N = X.shape[0]
        result_mat = np.zeros((N,self.n), float)
        result_mat[:, 0] = ((X >= 0) & (X <= 1)).astype(float) 
        #find  the translated input of psi
        y = X[:, None]*self.powers[None, :]  - self.ks[None, :]
        #result_mat is N by n
        #print(y)
        #print(self.powers_square_root[None, :])
        result_mat[:, self.l] =  self.powers_square_root[None, :] * self.psi(y) 
      
        return result_mat
    
#X=np.array([-0.01])
#mat= wavelet_1D(16,1).basis(X)
#mat
#mat[0]
#at 15, k=7, n=3 pih(8t-6)    
   


class generate_basis_mat:
    def __init__(self, n, dim,alpha):
        self.dim = dim
        self.n=n
        self.wavelet = wavelet_1D(n,alpha)
        inverse_alpha=1/alpha
        self.inverse_wavelet = wavelet_1D(n,inverse_alpha)

    def all_x_multivariate(self, X):

        N  = len(X)
        

        B = np.empty((N, self.dim, self.n), float)
        #print(self.wavelet.basis(X[:, 0]).shape)
        for j in range(self.dim):
        
            B[:, j, :] = self.wavelet.basis(X[:, j])
            
        return B

    def all_x_multivariate_inverse_alpha(self, X):
    
        N  = len(X)
        
    
        B = np.empty((N, self.dim, self.n), float)
        #print(self.wavelet.basis(X[:, 0]).shape)
        for j in range(self.dim):
        
            B[:, j, :] = self.inverse_wavelet.basis(X[:, j])
            
        return B
"""
alpha=0.3
mat=generate_basis_mat(n, dim,alpha=0.3).all_x_multivariate(data)

poly_basis=generate_basis_mat(n, dim,alpha=0.3)
basis_mat=np.array(poly_basis.all_x_multivariate(data))

mat.shape
diff=mat-basis_mat
np.sum(np.abs(diff))
###############


 
 
y=np.array([-1, 0.5])
mask = (y  >= 0) & ( y  < 1)
#frac = y - np.floor(y)
p = np.where(y< 0.5, 1.0, -1.0)
p[~mask] = 0.0
"""