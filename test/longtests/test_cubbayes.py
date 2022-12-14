import qmcpy as qp
import numpy as np
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor, kernels



import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data',header=None)
df.columns = ['Age','1900 Year','Axillary Nodes','Survival Status']
df.loc[df['Survival Status']==2,'Survival Status'] = 0
x,y = df[['Age','1900 Year','Axillary Nodes']],df['Survival Status']
xt,xv,yt,yv = train_test_split(x,y,test_size=.33,random_state=7)

print(df.head(),'\n')
print(df[['Age','1900 Year','Axillary Nodes']].describe(),'\n')
print(df['Survival Status'].astype(str).describe())
print('\ntrain samples: %d test samples: %d\n'%(len(xt),len(xv)))
print('train positives %d   train negatives: %d'%(np.sum(yt==1),np.sum(yt==0)))
print(' test positives %d    test negatives: %d'%(np.sum(yv==1),np.sum(yv==0)))
xt.head()

blr = qp.BayesianLRCoeffs(
    sampler = qp.DigitalNetB2(4,seed=7),
    feature_array = xt, # np.ndarray of shape (n,d-1)
    response_vector = yt, # np.ndarray of shape (n,)
    prior_mean = 0, # normal prior mean = (0,0,...,0)
    prior_covariance = 5) # normal prior covariance = 5I

qmc_sc = qp.CubQMCNetG(blr,
    abs_tol = .05,
    rel_tol = .5,
    error_fun = lambda s,abs_tols,rel_tols:
        np.minimum(abs_tols,np.abs(s)*rel_tols))
blr_coefs,blr_data = qmc_sc.integrate()
print(blr_data)

qmc_sc = qp.CubBayesNetG(blr,
    abs_tol = .05,
    rel_tol = .5,
    error_fun = lambda s,abs_tols,rel_tols:
        np.minimum(abs_tols,np.abs(s)*rel_tols))
blr_coefs,blr_data = qmc_sc.integrate()
print(blr_data)
# LDTransformData (AccumulateData Object)
#     solution        [-0.004  0.13  -0.157  0.008]
#     comb_bound_low  [-0.006  0.092 -0.205  0.007]
#     comb_bound_high [-0.003  0.172 -0.109  0.012]
#     comb_flags      [ True  True  True  True]
#     n_total         2^(18)
#     n               [[  1024.   1024. 262144.   2048.]
#                     [  1024.   1024. 262144.   2048.]]
#     time_integrate  2.229






##################################################################################

f = lambda x: np.cos(10 * x) * np.exp(.2 * x) + np.exp(-5 * (x - .4) ** 2)
xplt = np.linspace(0, 1, 100)
yplt = f(xplt)
x = np.array([.1, .2, .4, .7, .9])
y = f(x)
ymax = y.max()

gp = GaussianProcessRegressor(kernel=kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                              n_restarts_optimizer=16).fit(x[:, None], y)
yhatplt, stdhatplt = gp.predict(xplt[:, None], return_std=True)

tpax = 32
x0mesh, x1mesh = np.meshgrid(np.linspace(0, 1, tpax), np.linspace(0, 1, tpax))
post_mus = np.zeros((tpax, tpax, 2), dtype=float)
post_sqrtcovs = np.zeros((tpax, tpax, 2, 2), dtype=float)
for j0 in range(tpax):
    for j1 in range(tpax):
        candidate = np.array([[x0mesh[j0, j1]], [x1mesh[j0, j1]]])
        post_mus[j0, j1], post_cov = gp.predict(candidate, return_cov=True)
        evals, evecs = scipy.linalg.eig(post_cov)
        post_sqrtcovs[j0, j1] = np.sqrt(np.maximum(evals.real, 0)) * evecs


def qei_acq_vec(x, compute_flags):
    xgauss = scipy.stats.norm.ppf(x)
    n = len(x)
    qei_vals = np.zeros((n, tpax, tpax), dtype=float)
    for j0 in range(tpax):
        for j1 in range(tpax):
            if compute_flags[j0, j1] == False: continue
            sqrt_cov = post_sqrtcovs[j0, j1]
            mu_post = post_mus[j0, j1]
            for i in range(len(x)):
                yij = sqrt_cov @ xgauss[i] + mu_post
                qei_vals[i, j0, j1] = max((yij - ymax).max(), 0)
    return qei_vals


qei_acq_vec_qmcpy = qp.CustomFun(
    true_measure=qp.Uniform(qp.DigitalNetB2(2, seed=7)),
    g=qei_acq_vec,
    dimension_indv=(tpax, tpax),
    parallel=False)
qei_vals, qei_data = qp.CubQMCNetG(qei_acq_vec_qmcpy, abs_tol=.025, rel_tol=0).integrate()  # .0005
print(qei_data)

qei_vals, qei_data = qp.CubBayesNetG(qei_acq_vec_qmcpy, abs_tol=.025, rel_tol=0).integrate()  # .0005
print(qei_data)

a = np.unravel_index(np.argmax(qei_vals, axis=None), qei_vals.shape)
xnext = np.array([x0mesh[a[0], a[1]], x1mesh[a[0], a[1]]])
fnext = f(xnext)
