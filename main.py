import numpy as np





dim =50
N=dim*1000
N_test=5000
print('dim=', dim, ', N=', N, ', N_test=', N_test)

"""
######
###### beta mixture
from generate_beta_mixture import beta_mixture,conditional_sample_beta_mixture
alpha  =  [1*np.ones(dim),  3*np.ones(dim),2*np.ones(dim), ] 
alpha[0][::2]=2
beta  =  [3 *np.ones(dim),  2*np.ones(dim),1*np.ones(dim), ] 
#beta[0][::2]=2
weights= [0.4,0.3 ,0.3 ]
x_given=np.array([0.2, 0.3  ,0.5, 0.3 ])

dim_given=len(x_given)

generator= beta_mixture(dim,   alpha , beta , weights)
X_train= generator.generate(N)

X_test_conditional=conditional_sample_beta_mixture(alpha, beta, weights,  x_given, N_test)[:,dim_given:]

#X_test= generator.generate(N_test)
#y_test= generator.value(X_test)

"""

########
########sin (sum(x)) +1
########
########sin cos mixture
from generate_mixture_sin_cos import sin_cos_mixture
scale=2*np.pi
generator = sin_cos_mixture(dim, scale, scale,[0.5, 0.5])
X_train=generator.generate(N)
x_given=np.array([0.6, 0.3,0.1   ])
dim_given=len(x_given)
X_test_conditional=sin_cos_mixture(dim, scale, scale,[0.5, 0.5]).conditional_generate(x_given, N_test)[:,dim_given:]

X_test=generator.generate(N_test)
y_test=generator.value(X_test)


"""
from generate_sin_sum import sample_sin_sum,conditional_sample_sin_sum,sin_sum_density
x_given=np.array([0.4, 0.7 , 0.8, 0.2   ])
dim_given=len(x_given)
scale= 2*np.pi
X_train = sample_sin_sum(scale, dim, N)
X_test_conditional =conditional_sample_sin_sum (scale, dim,x_given, N_test)

X_test= sample_sin_sum(scale, dim, N_test)
y_test= sin_sum_density(scale, dim, X_test)
"""

#######################################
#######################################
#######################################
####################################### conditional sampling
from TT import TT 
ranks=[3 for _ in range(dim-1)]
n=10
s=300
alpha=1/np.sqrt(dim*n) 

TT_model= TT(n,ranks,alpha,s, X_train)




X_TT = TT_model.conditional_sample(np.array(x_given), N_test)

from Wasserstein import sliced_wasserstein_2 

e_TT=sliced_wasserstein_2(X_TT  , X_test_conditional)

print('TT conditional sampling error', e_TT   )

#XX_TT=TT_model.sample(N_test)
#print('TT Unconditional sampling error',sliced_wasserstein_2(XX_TT  , X_test))


#####################conditional sampling

from VAE import conditional_samples_cvae


X_vae = conditional_samples_cvae(
        X_train,
        x_given=x_given,
        N_sample=N_test,
        idx_given=None,     # assumes first 3 dims are given
        epochs=500,
        verbose=True
    )[:,dim_given:]


e_vae=sliced_wasserstein_2(X_vae  , X_test_conditional)

print('e_vae conditional sampling error', e_vae   )
#np.mean(X_vae, axis=0)




 
#######################################
#######################################
#######################################
####################################### Density Estimation
y_TT=TT_model.predict(X_test)
print('TT density estimation error (L_2)',np.linalg.norm(y_TT- y_test)/ np.linalg.norm(  y_test))

from kde import kernel_density
KDE= kernel_density(X_train)
y_kde=KDE.compute(X_test)
print('KDE density estimation error (L_2)',np.linalg.norm(y_kde- y_test)/ np.linalg.norm(  y_test))
