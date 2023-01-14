import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.stats import invwishart
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

# Read data from sheet 'B2:E289' of 'USdata_2019Q4.xlsx'
data = pd.read_excel('USdata_2019Q4.xlsx', sheet_name='Sheet1', usecols="B:E", header=None,skiprows=1)
Y = data.iloc[40:, [0,2]]
Y = np.asarray(Y)
Y0 = np.asarray(data.iloc[38:40, [0,2]])

# Define the number of lags
p = 2

# Define the number of variables
n = 2

# Define the number of observations
T = len(Y)



# Fit an AR(4) model to the pre-sample data
sample_variance = np.zeros(2)
pre_sample = np.asarray(data.iloc[:40, [0,2]])
model = sm.tsa.ARIMA(endog = pre_sample[:,0], order=(4,0,0))
results = model.fit()
# Extract the estimated residual variance
sample_variance[0] = float(results.summary().tables[1].data[-1][2])

pre_sample = np.asarray(data.iloc[:40, [0,2]])
model = sm.tsa.ARIMA(endog = pre_sample[:,1], order=(4,0,0))
results = model.fit()
# Extract the estimated residual variance
sample_variance[1] = float(results.summary().tables[1].data[-1][2])

# Define the Minnesota prior for beta
k_1 = 0.4**2
k_2 = 10**2
k_3 = 1

# Define the standard inverse-Wishart prior for Sigma
k_0 = n + 3
S_0 = k_3*np.eye(n)
nu_0 = 5
A_0 = np.zeros((n*p+1,2))

# Pre-compute matrices for the Minnesota prior for beta
X = np.ones((T, 1 + n*p))
X[0] = np.hstack((1, Y0[1], Y0[0]))
X[1] = np.hstack((1, Y[0], Y0[1]))
for i in range(T-2):
    X[i+2] = np.hstack((1, Y[i+1], Y[i]))

XTX = X.T@X
XTY = X.T@Y
YTY = Y.T@Y

# Define the strengh of shrinkage:
def compute_V_a(k_1,k_2,k_3, sample_variance):
    V_a = np.zeros(n*p+1)
    V_a[0] = k_2
    for i in range(n):
        for j in range(p):
            V_a[i*p + j +1] = k_1 / ((j+1)**2*sample_variance[i])
    return np.diag(V_a)
V_a = compute_V_a(k_1,k_2,k_3, sample_variance)

# Initialize the VAR coefficients and covariance matrix
Sigma = invwishart.rvs(df = k_0, scale = S_0)
beta = np.random.multivariate_normal(np.zeros(5), V_a, size = 2)


# Define the number of burn-in and sample iterations
Burn = 1000
Sample = 10000
N = Burn + Sample

import pdb; pdb.set_trace()
# Start the Gibbs sampling
Beta_tot = []
Sigma_tot = []
for g in range(N):

    # Draw beta from the posterior p(beta | y, Sigma)
    KA = np.linalg.inv(V_a) + XTX 
    A_hat = np.linalg.inv(KA)@(np.linalg.inv(V_a)@A_0 + XTY)
    beta = np.random.multivariate_normal(A_hat.reshape(10, order = 'f'), np.kron(Sigma, np.linalg.inv(KA)))

    # Draw Sigma from the posterior p(Sigma | y, beta)
    S_hat = S_0 + A_0.T@np.linalg.inv(V_a)@A_0 + YTY - A_hat.T@KA@A_hat
    Sigma = invwishart.rvs(df = nu_0 + T, scale = S_hat)
    # Save the samples
    # if g > Burn:
    Beta_tot.append(beta)
    Sigma_tot.append(Sigma)

Beta_tot = np.asarray(Beta_tot)
Sigma_tot = np.asarray(Sigma_tot)
import pdb; pdb.set_trace()
fig, ax = plt.subplots(5,2, sharey = True)
for i in range(5):
    for j in range(2):
        ax[i,j].hist(Beta_tot[:,i*2 +j])
plt.show()

