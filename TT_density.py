
import numpy as np



from wavelet import generate_basis_mat


from TT_ultility import TT_prediction
from transform_domain import domain





class hard_threshold:
    def __init__(self, N,dim,n,ranks,alpha,data) -> None:
        self.N=N
        self.dim=dim
        self.n=n
        self.ranks=ranks

        self.core_set=[]
        #self.core_set[d] is the d-th core of size (rank[d-1], n, rank[d]) 

        basis_fun=generate_basis_mat(n, dim,alpha)
        self.basis_mat= basis_fun.all_x_multivariate(data)

     
    def compute_first_core(self):
        current_index=0
        next_index=1
         
        
        flat_temp= self.basis_mat[:, next_index:,1:]
        moments= flat_temp.reshape(self.N, -1)
        moments = np.concatenate([moments, np.ones((self.N, 1))], axis=1) 
        A_temp =self.basis_mat[:,current_index,:]
        
        target_mat= A_temp.T @  moments
        
        
        U_0= np.linalg.svd(target_mat , full_matrices=False)[0]
        G_0=U_0.transpose()[:self.ranks[0]]
        G_0=G_0.transpose()
        self.core_set.append(G_0)
        #G_0 is size (n, rank[0])
        ######### update of tensor
        
        
        B_temp =self.basis_mat[: , current_index, :] @ G_0 
        C_temp = self.basis_mat[:, next_index,:]
        self.cur_basis  = (B_temp[:, :, None] * C_temp[:, None, :]).reshape(self.N, -1)
        
        
        return G_0, self.cur_basis
    

    def compute_second_to_last_core(self):
        self.compute_first_core()
        
        for j in range(1, self.dim-1):
            print(j)
            target_mat=[]
            next_index=j+1
            flat_temp= self.basis_mat[:, next_index:,1:]
            moments= flat_temp.reshape(self.N, -1)
            moments = np.hstack([moments, np.ones((self.N, 1))])
            target_mat = self.cur_basis.T @  moments 
            

            U_cur= np.linalg.svd(target_mat, full_matrices=False)[0]
            G_cur=U_cur.transpose()[:self.ranks[j]]
            G_cur=G_cur.transpose()
            
            #G_cur is shape (ranks[j-1]*n ,ranks[j])
            G_cur_tensor =G_cur.reshape(self.ranks[j-1],self.n,self.ranks[j])
            self.core_set.append(G_cur_tensor)

            #self.core_set[d] is shape (ranks[d-1],n, ranks[d])
            #self.all_coefficient_mat[i][d] is shape (ranks[d-1],n)
            #the updated version of self.all_coefficient_mat[i][d+1] is size (ranks[d],n)
            
            B_temp =self.cur_basis @ G_cur 
            C_temp = self.basis_mat[:, next_index,:]
            self.cur_basis  = (B_temp[:, :, None] * C_temp[:, None, :]).reshape(self.N, -1)
            
            
            
            
        G_temp =np.array(self.cur_basis).sum(axis=0) 
        self.core_set.append(G_temp.reshape(self.ranks[self.dim-2],self.n) /self.N  )
        
        return self.core_set



class TT_svd:
    def __init__(self, N,dim,n,ranks,alpha,X_train) -> None:
        
 
        self.n=n
        self.dim=dim
        self.alpha=alpha
        self.new_domain=domain(dim, X_train)
        X_train_transform=self.new_domain.compute_data(X_train)


        self.core_set=hard_threshold( N,dim,n,ranks,alpha,X_train_transform).compute_second_to_last_core()
    
    def compute(self, X_test):

        X_test_transform= self.new_domain.compute_data(X_test)
        

        y_TT=[]
        
        basis_function=generate_basis_mat(self.n, self.dim, self.alpha)

        mat_test= basis_function.all_x_multivariate_inverse_alpha(X_test_transform)
        for vec in mat_test:
            temp = TT_prediction().predict(self.dim, self.core_set,vec)
            y_TT.append(self.new_domain.transform_density_val(temp ))

        return y_TT 


