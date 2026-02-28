import numpy as np
from polynomial import generate_basis_mat,polynomial
from scipy.linalg import eigh



#################
#################
#################
#################
################# Using Nystrom to compute cores
class form_nystrom_matrix:
    def __init__(self,dim,N,n,s):
        self.dim=dim
        self.N = N
        self.n = n
        self.s =s 
    def compute(self, basis_xy):
        
        dp_M= [[] for __ in range(self.dim)]
        dp_W= [[] for __ in range(self.dim)]
        I_index =  np.random.choice(self.N, size=min(self.s,self.N), replace=False)
        temp= basis_xy[self.dim-1][I_index,:] 
        dp_M[self.dim-1] = basis_xy[self.dim-1] @ temp.T
        dp_W[self.dim-1]= temp@temp.T
        
        for j in range(self.dim-2,0,-1):
            temp= basis_xy[j][I_index,:] 
            
            dp_M[j]= dp_M[j+1]* (basis_xy[j] @ temp.T)
            dp_W[j]= dp_W[j+1] * (temp@temp.T)
        return dp_M, dp_W





class nystrom:
    def __init__(self, N,dim,n ,ranks,alpha,s,X_train ) -> None:
        self.N=N
        self.dim=dim
        self.n=n
        #ranks are size dim vector and ranks[j] is the rank of the j-th coordinate
        self.ranks=ranks

        self.core_set=[]
        #self.core_set[j] is the j-th core of size (rank[j-1], n, rank[j]) 
        ## basis_mat[i][j][k] corresponds to  phi_k(X_i(j)). 
        
        
        self.basis_mat =  generate_basis_mat(n, dim,alpha,X_train).compute().transpose(1, 0, 2) 
        
        

        self.dp_M, self.dp_W = form_nystrom_matrix(dim, N, n,s).compute(self.basis_mat)
        self.basis_mat=list(self.basis_mat)
    def all_cores(self):
         
        
        for j in range(0,self.dim-1):
            #compute the symmetric targeted matrix
            #print('coordinate',j)
            target_mat= self.basis_mat[j].T @ self.dp_M[j+1] @  np.linalg.pinv( self.dp_W [j+1], rcond=1e-8) @self.dp_M[j+1].T @self.basis_mat[j]
            
            target_mat= -1*(target_mat+ target_mat.T)*0.5
            _, G_cur = eigh(target_mat, subset_by_index=[0, self.ranks[j]-1])
            #G_cur=np.linalg.svd(target_mat , full_matrices=False, hermitian=True)[0][:,:self.ranks[j]]

             
            
            if j==0:
                G_cur_tensor= G_cur
            if j>0:
                #G_cur is shape (ranks[j-1]*n ,ranks[j])
                G_cur_tensor =G_cur.reshape(self.ranks[j-1],self.n,self.ranks[j])
            self.core_set.append(G_cur_tensor)
            #self.core_set[j] is shape (ranks[j-1],n, ranks[j ])
            
            
            
            #print(self.basis_mat[0][j].shape)
            #i=0
            #print(np.tensordot(G_cur,self.basis_mat[0][j],axes=(0,0)))
            #print(np.outer( np.tensordot(G_cur,self.basis_mat[i][j],axes=(0,0)), self.basis_mat[i][j+1] ).reshape(-1))
            #G_cur is shape ( ranks[j-1]*n ,rank[j] )
            #self.basis_xy[i][j ] is shape (N, ranks[j-1]*n )
            #the updated version of self.basis_xy[j+1] is  shape (N, ranks[j]*n)
            
            A_temp =  self.basis_mat[j] @ G_cur
            B_temp = self.basis_mat[j+1]
            
            self.basis_mat[j+1] = (A_temp[:, :, None] * B_temp[:, None, :]).reshape(self.N, -1)
        
        #print(self.ranks)
        ###last core
        G_temp =self.basis_mat[self.dim-1].sum(axis=0) 
        self.core_set.append(G_temp.reshape(self.ranks[self.dim-2],self.n)  /self.N  )

        return self.core_set 


#################
#################
#################
################# compute density value
from  sampling_1D import Legendre_Sampler



class TT:
    def __init__(self ,n,ranks,alpha,s,X_train) -> None:
        
 
        self.n=n
        N, self.dim=X_train.shape[0], X_train.shape[1]
        self.alpha=alpha
        self.new_domain=domain(self.dim, X_train)
        self.X_train_transform=self.new_domain.transform_data(X_train)
        self.X_train_transform=np.clip(self.X_train_transform, 0, 1)

        self.count=0
        self.core_set=nystrom (  N,self.dim,n ,ranks,alpha,s,self.X_train_transform ).all_cores()
        
        inv_vec=(alpha**-1)*np.ones(n)
        inv_vec[0]=1
        #core_set[0][:,1]

        self.core_set[0]= inv_vec[:,None]*self.core_set[0]
        for d in range(1,self.dim-1):
            self.core_set[d]= self.core_set[d]* inv_vec[None,:, None]
        self.core_set[-1]= inv_vec[None,:]*self.core_set[-1]
        self.marginal_mat=[[] for __ in range(self.dim)]
        #self.marginal_mat is shape (d,r)
        self.marginal_mat[-1] = self.core_set[-1][:,0]
        
        for d in range(self.dim-2,0,-1 ):
            #print(d)
            self.marginal_mat[d] = self.core_set[d][:,0,:]@self.marginal_mat[d+1]
        #print(self.marginal_mat)
        self.sampler=Legendre_Sampler(n)
        self.polynomial = polynomial(n, 1)
        #print(self.core_set[0])

    def conditional_sampling_one(self,  c_dim ):
        
        result=np.zeros(self.dim)
        
        cur_condition_mat= self.condition_mat
        for d in range(c_dim, self.dim-1):
            cur_vec = np.einsum('r, rns, s->n',  cur_condition_mat,self.core_set[d], self.marginal_mat[d+1])
            result[d]= self.sampler.sample(cur_vec)
            if result[d]==-np.inf:
                
                return False,[]
            cur_basis_val= self.polynomial.scalar_all_basis_alpha(result[d])
           
            cur_condition_mat= np.einsum('r, rns, n ->s', cur_condition_mat, self.core_set[d],cur_basis_val)
        
        
        ###last coordinates
        cur_vec = np.einsum('r, rn ->n',  cur_condition_mat,self.core_set[-1] )
        result[-1]= self.sampler.sample(cur_vec) 
        if result[-1]==-np.inf:
            
            return False,[]
        return True, result[c_dim:]
    
    def conditional_sample(self,x_given, N_sample):
        c_dim= x_given.shape[0]
        x_given_transform= self.new_domain.transform_partial_data(0,c_dim, x_given)
        X_conditional_TT=np.zeros((N_sample, self.dim-c_dim))
        
        cur_basis_val= self.polynomial.scalar_all_basis_alpha(x_given_transform[0])
        #condition_mat is (r,1)
        self.condition_mat= np.einsum('n, nr->r', cur_basis_val, self.core_set[0])
        ####Given coordinates
        for d in range(1, c_dim):
            
            
            cur_basis_val= self.polynomial.scalar_all_basis_alpha( x_given_transform[d])
            #condition_mat is (r,1)
            self.condition_mat= np.einsum('r, rns, n ->s', self.condition_mat, self.core_set[d],cur_basis_val)
        
        
        for i in range(N_sample):
            if i%100==0:
                print(i)
            rec, temp =  self.conditional_sampling_one(  c_dim) 
            while not rec :
                self.count+=1
                print('count',self.count)
                rec,temp =  self.conditional_sampling_one(  c_dim ) 
            X_conditional_TT[i]= temp
            #if np.isneginf(X_conditional_TT[i]).all():
            #    print('error',i)
        #print(X_conditional_TT)
        X_conditional_TT=self.new_domain.inverse_partial_data(c_dim,self.dim,X_conditional_TT)
        return X_conditional_TT 
    
    def sample_one(self):
        
        ####first_coordinate
        result=np.zeros(self.dim)
        cur_vec=self.core_set[0]@self.marginal_mat[1]
        #result[0]= self.sampler.sample(cur_vec,    )[0]
        result[0]= self.sampler.sample(cur_vec) 
        cur_basis_val= self.polynomial.scalar_all_basis_alpha(result[0])
        #print(cur_basis_val)
        #condition_mat is (r,1)
        condition_mat= np.einsum('n, nr->r', cur_basis_val, self.core_set[0])
        ####Middle coordinates
        for d in range(1, self.dim-1):
            cur_vec = np.einsum('r, rns, s->n',  condition_mat,self.core_set[d], self.marginal_mat[d+1])
            result[d]= self.sampler.sample(cur_vec) 
            if result[d]==-np.inf:
                
                return False,[]
            cur_basis_val= self.polynomial.scalar_all_basis_alpha(result[d])
            #condition_mat is (r,1)
            condition_mat= np.einsum('r, rns, n ->s', condition_mat, self.core_set[d],cur_basis_val)
        
        
        ###last coordinates
        cur_vec = np.einsum('r, rn ->n',  condition_mat,self.core_set[-1] )
        result[-1]= self.sampler.sample(cur_vec) 
        if result[-1]==-np.inf:
            
            return False,[]
        return True,result
    
    def sample(self, N_sample):
    
        result=np.zeros((N_sample, self.dim))
        for i in range(N_sample):
            if i%100==0:
                print(i)
            rec, temp =  self.sample_one(  ) 
            while not rec :
                rec, temp =  self.sample_one(  )
            result[i]= temp       
        X_TT=self.new_domain.inverse_compute_data(result)
        return X_TT

    def predict(self, X_test):

        X_test_transform= self.new_domain.transform_data(X_test)
        X_test_transform=np.clip(X_test_transform, 0, 1)

        y_TT=[]
        
        

        mat_test= generate_basis_mat(self.n, self.dim,1, X_test_transform).compute()
        y_TT = TT_prediction().predict(self.dim,  self.core_set , mat_test)
        y_TT= self.new_domain.transform_density_val(y_TT)
        #for vec in mat_test:
        #    temp = TT_prediction().predict(self.dim, self.core_set,vec)
        #   y_TT.append(self.new_domain.transform_density_val(temp ))

        return np.clip(y_TT, 1e-14, np.inf) 


    
#################
#################
#################
################# domain related


    
    

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
        self.upper,self.lower = self.upper+ 0.001* (self.upper-self.lower),self.lower- 0.001* (self.upper-self.lower)

        #print(self.upper)
        #print(self.lower)
        self.difference =self.upper-self.lower
        self.density_factor=np.prod(self.difference)
        
    def transform_density_val(self, val):
        return val/self.density_factor
        



    def transform_data(self,XX):
        X_transform= (XX-self.lower)/self.difference
        return  X_transform 
    def transform_partial_data(self,begin_dim, end_dim,XX):
        X_transform= (XX-self.lower[begin_dim:end_dim])/self.difference[begin_dim:end_dim]
        return  X_transform 

    
    def inverse_compute_data(self, UU):
        UU = np.asarray(UU)
        return UU * self.difference + self.lower    
    
    def inverse_partial_data(self, begin_dim, end_dim,UU):
        UU = np.asarray(UU)
         
        return UU * self.difference[begin_dim:end_dim] + self.lower[begin_dim: end_dim]  
    
    
    
#######prediction related
#################
#################
#################
class TT_prediction:
    def __init__(self):
        pass

    def predict(self, dim, core_set, vec_set):
        """
        core_set[0]: shape (n, r0)
        core_set[d]: shape (r_{d-1}, n, r_{d}) for 0 <= d < dim-1
        core_set[dim-1]: shape (r_{dim-2}, n)
        vec_set: shape (N, dim, n)
        returns: array of shape (N,)
        """

        # 1) First contraction: (N, n) x (n, r1) -> (N, r1)
        #    vec_set[:,0,:] has shape (N,n)
        #    core_set[0]   has shape (n,r1)
        res = np.tensordot(vec_set[:, 0, :], core_set[0], axes=(1, 0))
        # res.shape == (N, r1)

        # 2) Loop over intermediate cores 1 .. dim-2
        #    Each step: res (N, r_d), vec_set[:,d,:] (N, n), core_set[d] (r_d, n, r_{d+1})
        #    We want new res of shape (N, r_{d+1})
        for d in range(1, dim-1):
            # use einsum to batch over N
            # 'Nr, Nn, rnk -> Nk' means:
            #    for each sample i: res[i,r] * vec_set[i,n] * core_set[r,n,k]
            res = np.einsum('Nr,Nn,rnk->Nk',
                            res,
                            vec_set[:, d, :],
                            core_set[d])

        # 3) Final contraction with last core core_set[dim-1] of shape (r_{D-1}, n)
        #    and vec_set[:, dim-1, :] of shape (N, n)
        #    so (N, r_{D-1}) x (r_{D-1}, n) x (N, n) -> (N,)
        final_core = core_set[dim-1]
        res = np.einsum('Nr,rn,Nn->N', res, final_core, vec_set[:, dim-1, :])

        return res


 
