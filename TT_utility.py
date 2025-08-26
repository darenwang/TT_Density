
import numpy as np






###### Given ranks of size dim
###### Find TT decomposition from the full tensor
###### return tensor cores


class full_tensor_TT:
    def __init__(self):
        self.core_set=[]
        
    def compute(self, dim,n,ranks, full_tensor):

        reshape_cur_mat=full_tensor.reshape(n,-1)

        U_1 = np.linalg.svd(reshape_cur_mat, full_matrices=False)[0]
        G_1=U_1.transpose()[:ranks[0]]
        G_1=G_1.transpose()
        self.core_set.append(G_1)
        cur_mat=np.tensordot(self.core_set[-1],full_tensor,axes=(0,0))


        for d in range(1,dim-1):
            
            #print(cur_mat)
            reshape_cur_mat=cur_mat.reshape(ranks[d-1]*n, -1)
            U_cur= np.linalg.svd(reshape_cur_mat, full_matrices=False)[0]
            G_cur=U_cur.transpose()[:ranks[d]]
            G_cur=G_cur.transpose()
            G_cur=G_cur.reshape(ranks[d-1],n,ranks[d])
            self.core_set.append(G_cur)
            #print(self.core_set[-1].shape, cur_mat.shape)
            cur_mat=np.tensordot(self.core_set[-1],cur_mat, axes=([0,1],[0,1]))

        self.core_set.append(cur_mat[:])
        return self.core_set



class TT_prediction:
    def __init__(self):
        pass
    def predict(self, dim, core_set, vec_set):
        left  =vec_set[0] @ core_set[0]
        for d in range(1,dim-1):
            #left is size r
            M = np.einsum('anb, n-> ab', core_set[d], vec_set[d])
            left=  left@M
        #print(temp.shape,new_core.shape)
        result= left@core_set[dim-1]@vec_set[dim-1]
        return result
"""

class TT_prediction:
    def __init__(self):
        pass
    def predict(self, dim, core_set, vec_set):
        new_core=core_set[0]
        for d in range(dim-1):
            #temp is size r
            #new_core is size n times r
            #vec_set[d] is size (n,)
            #core_set[d+1] is r times n times r
            temp=np.tensordot([vec_set[d]],new_core,axes=(1,0))

            new_core= np.tensordot(temp,core_set[d+1],axes=(1,0))[0]
        #print(temp.shape,new_core.shape)
        temp=np.tensordot([vec_set[dim-1]],new_core,axes=(1,0))
        return temp[0]

"""