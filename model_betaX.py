import numpy as nps
def generate_data(n_samples):
    RANDOM_SEED = 8927
    rng = nps.random.default_rng(RANDOM_SEED)
    alpha = 0.25
    sigma_e = 1
    betas = [-1, 1.5]
    mean = [0.5, 0.75, 1.5]
    sigmas = [1, 0.7**0.5, 1.2]
    cov = [[sigmas[0]**2, 0, 0], [0, sigmas[1]**2, 0.7], [0, 0.7, sigmas[2]**2]]
   
    X_samples = nps.random.multivariate_normal(mean=mean, cov=cov, size=n_samples)
   
    y = alpha + betas[0] * X_samples[:, 0]  + betas[1] * X_samples[:, 1] + rng.normal(size=n_samples) * sigma_e
   
    return y, X_samples

import matplotlib.pyplot as plt
import jax.numpy as np
from jax import grad, jit, random
# Mean of the distribution
key = random.PRNGKey(0)
# Covariance matrix of the distribution
alpha = 0.25

Y, X_samples = generate_data(50)
Y = np.array(Y)
X1 = np.array(X_samples[:,0])
X2 = np.array(X_samples[:,1])
X3 = np.array(X_samples[:,2])


def likelihood(alpha, beta_1, beta_2, sigma, X1 = X1, X2 = X2, Y = Y):
    return np.sum((-0.5/sigma**2*(np.linalg.norm(Y-alpha-beta_1*X1-beta_2*X2, axis = 0)) ** 2)) - len(Y)*np.log(sigma*np.sqrt(2*np.pi))

def prior_alpha(mu):
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    return random.normal(subkey) + mu

def prior_beta(sigma):
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    return random.normal(subkey)*sigma

def generate_samples(mu, sigma, sigma_beta_1, sigma_beta_2, X1 = X1, X2 = X2, Y = Y):
    N = 1000
    n = len(Y)
    beta_1 = np.zeros(N)
    beta_2 = np.zeros(N)
    alphas = np.zeros(N)

    burn_in = 0
    alpha_tmp = prior_alpha(mu)
    beta_tmp1 = prior_beta(sigma_beta_1)
    beta_tmp2 = prior_beta(sigma_beta_2)
    for i in range(N+burn_in):
        key = random.PRNGKey(0)
        key, subkey = random.split(key)
        alpha_tmp = random.normal(subkey)*np.sqrt((sigma**2/(n+sigma**2))) + (np.sum(Y-beta_tmp1*X1 - beta_tmp2*X2) + sigma**2*mu)/(n + sigma**2)
        if i > burn_in:
            alphas = alphas.at[i-burn_in].set(alpha_tmp)

        
        key, subkey = random.split(key)
        beta_tmp1 = random.normal(subkey)*np.sqrt((sigma*sigma_beta_1)**2/(sigma**2 +np.mean(X1**2)*sigma_beta_1**2)) + sigma**2*np.sum(X1*(Y-alpha_tmp-beta_tmp2*X2))/(sigma**2 + np.sum(X1**2)*sigma_beta_1**2)
        if i > burn_in:
            beta_1 = beta_1.at[i-burn_in].set(beta_tmp1)


        key, subkey = random.split(key)
        beta_tmp2 = random.normal(subkey)*np.sqrt((sigma*sigma_beta_2)**2/(sigma**2 +np.mean(X2**2)*sigma_beta_2**2)) + sigma**2*np.sum(X2*(Y-alpha_tmp-beta_tmp1*X1))/(sigma**2 + np.sum(X2**2)*sigma_beta_2**2)
        if i > burn_in:
            beta_2 = beta_2.at[i-burn_in].set(beta_tmp2)
    return alphas, beta_1, beta_2


def Chibs_method(param_dict, samples = Y):
    alpha_samples = np.array(generate_samples(mu = param_dict['mu'], sigma = param_dict['sigma'], sigma_beta_1 = param_dict['sigma_beta_1'], sigma_beta_2 = param_dict['sigma_beta_2']))

    psi_etoile = np.mean(alpha_samples, axis = 0)
    var_psi = np.var(alpha_samples, axis = 0)
    psi_etoile_moins, psi_etoile_plus = psi_etoile - .5, psi_etoile + .5

    proba_psi = np.sum((alpha_samples> psi_etoile_moins)*(alpha_samples < psi_etoile_plus), axis = 0)/len(alpha_samples)

    log_proba = np.sum(np.log(proba_psi))
    log_prior = -0.5*(np.linalg.norm(psi_etoile))**2
    log_lik = likelihood(psi_etoile[0], psi_etoile[1], psi_etoile[2], param_dict['sigma'])

    log_marg = log_lik + log_prior - log_proba
    
    return log_marg

test = generate_samples(0.25, 0.1, 0.1, 0.1, X1, X2, Y)
# import numpy
# test = numpy.asarray(test)
# numpy.savetxt("values_with_X3.csv", test, delimiter=",")
test_grad = grad(Chibs_method)
param_dict = {'mu': 0.25, 'sigma': 0.1, 'sigma_beta_1': 0.1, 'sigma_beta_2' : .1}
test_2 = test_grad(param_dict)
# fig, ax = plt.subplots(3,1, sharey = True)
# for i in range(3):
#     ax[i].hist(test[i])
# plt.show()
import pdb; pdb.set_trace()