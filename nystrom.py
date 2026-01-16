import numpy as np
from polynomial import generate_basis_mat
from scipy.linalg import eigh
from transform_domain import domain
from TT_utility import   TT_prediction


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
        
        
        basis_mat=generate_basis_mat(n, dim,alpha,X_train).compute()
        self.basis_mat =  basis_mat.transpose(1, 0, 2).copy()
        
        

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

class TT_svd:
    def __init__(self ,n,ranks,alpha,s,X_train) -> None:
        
 
        self.n=n
        N, self.dim=X_train.shape[0], X_train.shape[1]
        self.alpha=alpha
        self.new_domain=domain(self.dim, X_train)
        self.X_train_transform=self.new_domain.compute_data(X_train)
        self.X_train_transform=np.clip(self.X_train_transform, 0, 1)

        
        self.core_set=nystrom (  N,self.dim,n ,ranks,alpha,s,self.X_train_transform ).all_cores()
        #print(self.core_set[0])
    def predict(self, X_test):

        X_test_transform= self.new_domain.compute_data(X_test)
        X_test_transform=np.clip(X_test_transform, 0, 1)

        y_TT=[]
        
        

        mat_test= generate_basis_mat(self.n, self.dim,self.alpha**-1, X_test_transform).compute()
        y_TT = TT_prediction().predict(self.dim,  self.core_set, mat_test)
        y_TT= self.new_domain.transform_density_val(y_TT)
        #for vec in mat_test:
        #    temp = TT_prediction().predict(self.dim, self.core_set,vec)
        #   y_TT.append(self.new_domain.transform_density_val(temp ))

        return np.clip(y_TT, 1e-14, np.inf) 

from sklearn.model_selection import train_test_split

class nystrom_cv:
    def __init__(self, N,dim  ,X_train) -> None:
        
        
        #ranks are size dim vector and ranks[j] is the rank of the j-th coordinate
        self.X=X_train
        self.X1, self.X2 = train_test_split(X_train, test_size=0.5, random_state=0, shuffle=True)

        self.core_set=[]
        #self.core_set[j] is the j-th core of size (rank[j-1], n, rank[j]) 
        ## basis_mat[i][j][k] corresponds to  phi_k(X_i(j)). 
        
    
    def compute(self,set_parameters ):
        cur_log_likelihood=-np.inf
        cur_candidate=-1
        for kk in range(len(set_parameters)):
            
            ranks,n,alpha,s = set_parameters[kk]
            
            y_TT_cv= TT_svd( n,ranks,alpha,s,self.X1).predict(self.X2)
            
            temp_log = np.mean(np.log(  y_TT_cv   )  )
            if temp_log > cur_log_likelihood:
                cur_candidate=kk
                cur_log_likelihood=temp_log
            #print(kk, temp_log)
        #print('select', cur_candidate)
        cv_ranks, cv_n, cv_alpha, cv_s = set_parameters[cur_candidate]
        
        TT_model= TT_svd( cv_n,cv_ranks,cv_alpha,cv_s,self.X) 
        
        return  TT_model
        
    
    