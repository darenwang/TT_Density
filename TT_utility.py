
import numpy as np






###### Given ranks of size dim
###### Find TT decomposition from the full tensor
###### return tensor cores


class full_tensor_TT:
    def __init__(self):
        self.core_set=[]
        
    def compute_svd(self, dim,n,ranks, full_tensor):

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

    def full_TT(self,core_set):
        rec=core_set[0]
        for j in range(1,len( core_set)):
            rec= np.tensordot(rec,  core_set[j], axes=([-1],[0]))
        return rec




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
