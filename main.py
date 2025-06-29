
import numpy as np
from gaussian_mixture import gaussian_mixture
from density_ultility import KL, kernel_density
from TT_density import TT_svd


N=10**5
dim=50
n=32
N_test=10**4



#alpha=1


print( 'sample size=', N)
print( 'dim=', dim)
print( 'n=', n)



means=[0, 0.05]
standard_deviations=[0.1,0.1]
generator = gaussian_mixture(dim, means, standard_deviations )
X_train=generator.generate(N)
X_test=generator.generate(N_test)

"""
y_true stores the true density values at the test data.
It is crucial to check y_true before running the code.
In 50 dimensions, for example, when the standard_deviations are larger than 0.2, 
Entries of y_true are likely to be less than 10**-10 or even smaller.
In this case, due to floating-point arithmetic, Python’s default precision is 6 digits.
So it is extremely hard to compute y_true accurately, even if the true density function is known.
Need to consider log transform when dealing with values <10**-10.
"""
############################### compute TT prediction at test data
ranks=[2 for _ in range(dim-1)]
alpha=1/np.sqrt(dim*n) 


TT_model= TT_svd(N,dim,n,ranks,alpha,X_train)
y_TT = TT_model.compute(X_test)


############## compute true density value at test data



y_true=[generator.value(x) for x in X_test ] 
y_true=np.array(y_true)



############# compute KL divergence
y_TT =np.array(y_TT)

TT_error=  KL.compute(y_TT, y_true)

print('TT KL=',TT_error)
###### L_2 error is meaningless due to the curse of dimensionality
#print(np.linalg.norm(y_TT_transform-y_true)  

################ compute KDE prediction at test data


kde=kernel_density(X_train)

y_kde= kde.compute(X_test)
y_kde=np.array(y_kde)

kde_error=  KL.compute(y_kde, y_true)
print("KDE KL=" ,  kde_error)


