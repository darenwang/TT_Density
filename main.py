
import numpy as np
from gaussian_mixture import gaussian_mixture
from density_utility import  kernel_density


N=10**5
dim=50
N_test=10**4



#alpha=1


print( 'sample size=', N)
print( 'dim=', dim)




means=[-0.5, 0.5]
standard_deviations=[1,1]
generator = gaussian_mixture(dim, means, standard_deviations )








from nystrom import TT_svd
ranks=[2 for _ in range(dim-1)]
n=15
alpha=1/np.sqrt(dim)/n 
s=300
for __ in range(20):
    X_train=generator.generate(N)
    X_test=generator.generate(N_test)
    y_true=generator.value(X_test)
    
    
    
    ###############################  TT prediction via Nystrom 

    y_TT= TT_svd(N, dim, n, ranks, alpha, s, X_train).predict(X_test)
    
    
    print( 'TT error',  np.linalg.norm(y_TT  -y_true)/np.linalg.norm( y_true)) 

    ########## KDE  very slow when N is large 
    kde=kernel_density(X_train)

    y_kde= kde.compute(X_test)


    y_kde=np.array(y_kde)
    kde_error= np.linalg.norm(y_kde-y_true)/ np.linalg.norm(y_true)
    print('kde error', kde_error)

#####################################
