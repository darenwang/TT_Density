
import numpy as np
import pandas as pd

ppp = pd.read_csv("ppp.csv")
cols = ["start_lon", "start_lat", "tor_length", "tor_width","end_lat",
"end_lon"]
data_locations = ppp[[c for c in cols if c in ppp.columns]].copy()

data=np.array(data_locations)

from distance import  sliced_wasserstein_2


train_min = data.min(axis=0)
train_max = data.max(axis=0)
denom= np.maximum(train_max - train_min, 1e-12)
data = (data - train_min) / denom

from sklearn.model_selection import train_test_split


X_train, X_test  = train_test_split( data   , test_size=0.3)
N=X_test.shape[0]
#N_test=X_test.shape[0]

n=20



#from Tucker   import Tucker
#TT_model= Tucker(n ,  X_train  ,10,ranks)

from tucker_hooi import Tucker
TT_model= Tucker(n ,  X_train  ,10,threshold=0.01,rank_rule="relative", )

print('y')
#y_TT=TT_model.predict(X_test)
X_TT=TT_model.sample(N )
#print('TT',sliced_wasserstein_2(X_test, X_TT))


from kde import kernel_density

kde= kernel_density(X_train)

X_kde=kde.sample(N)






from VAE import unconditional_samples_vae

X_vae= unconditional_samples_vae(X_train, N ,verbose=True)


np.save("X_vae.npy", X_vae)
X_vae = np.load("X_vae.npy")













from diffusion import TorchCFMTabularGenerator
diff = TorchCFMTabularGenerator(
    hidden_dims=(128, 128, 128),
    lr=3e-4,
    batch_size=256,
    epochs=200,
    sigma=0.03,
    standardize=True,
    weight_decay=1e-4,
    grad_clip=1.0,
    use_ema=True,
    ema_decay=0.9995,
    sample_solver="rk4",
    sample_steps=128,
    device=None
).fit(X_train)

X_diff = diff.sample(N , chunk_size=10000)


print('TT',sliced_wasserstein_2(X_test, X_TT))

print('kde',sliced_wasserstein_2(X_test, X_kde))

print('vae' ,sliced_wasserstein_2( X_test,X_vae))


print('diff',sliced_wasserstein_2( X_test,X_diff))


from heat import four_heatmaps

dim_1= 0
dim_2= 1


four_heatmaps(X_test, X_TT, X_vae, X_diff,
dim_1, dim_2,
bins=100,
sigma=1,
xlim=(-0.2, 1),
ylim=(-0.2, 1),
vmin=0,
vmax=15)


from heat import four_heatmaps

dim_1= 2
dim_2= 3


four_heatmaps(X_test, X_TT, X_vae, X_diff,
dim_1, dim_2,
bins=100,
sigma=1,
xlim=(-0.2, 1),
ylim=(-0.2, 1),
vmin=0,
vmax=25)