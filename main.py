
import numpy as np
from gaussian_mixture import gaussian_mixture
from density_utility import KL, kernel_density
from TT_density import TT_svd


N=10**5
dim=20
n=32
N_test=10**4



#alpha=1


print( 'sample size=', N)
print( 'dim=', dim)
print( 'n=', n)



means=[0, 0.3]
standard_deviations=[0.3,0.3]
generator = gaussian_mixture(dim, means, standard_deviations )
X_train=generator.generate(N)
X_test=generator.generate(N_test)


############################### compute TT prediction at test data
ranks=[2 for _ in range(dim-1)]
alpha=1/np.sqrt(dim*n) 


TT_model= TT_svd(N,dim,n,ranks,alpha,X_train)
y_TT = TT_model.compute(X_test)


############## compute true density value at test data



y_true = [generator.value(x) for x in X_test ] 
y_true = np.array(y_true)



############# compute KL divergence
y_TT =np.array(y_TT)

TT_error=  KL.compute(y_TT, y_true)

print('TT KL=',TT_error)
###### L_2 error is meaningless due to the curse of dimensionality
print( "TT relative l_2 norm", np.linalg.norm(y_TT-y_true)  / np.linalg.norm(y_true))

################ compute KDE prediction at test data


kde=kernel_density(X_train)

y_kde= kde.compute(X_test)
y_kde=np.array(y_kde)

kde_error=  KL.compute(y_kde, y_true)
print("KDE KL=" ,  kde_error)
print("KDE relative l_2 norm", np.linalg.norm(y_kde-y_true)  / np.linalg.norm(y_true))
