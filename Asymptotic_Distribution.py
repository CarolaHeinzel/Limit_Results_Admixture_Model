import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd


# Load the data
data_p = "example_data.P"
data_q = "example_data.Q"

df_q = pd.read_csv(data_q, header = None, delim_whitespace=True,dtype=float)
df_p = pd.read_csv(data_p, header=None, delim_whitespace=True, dtype=float)

K = len(df_p.iloc[0,]) # Number of populations
M = len(df_p) # Number of markers
#%%

# Implementation of the Central Limit Theorem
def res(q0, p1, uc):
    theta1 = np.dot(q0, p1)
    p1 = np.array(p1)
    if(uc ==1):
        p = np.array(p1 - p1[-1])
        p2 = np.outer(p[:-1],p[:-1])
    else:
        p = np.array(p1)
        p2 = np.outer(p,p)
    if(theta1 > 0 and theta1 < 1):
        res4 = 2*p2*(1/theta1) + 2*p2/(1-theta1)
    else:
        res4 = 2*p2

    return res4

def res_alle(M, q, p):
    res_int = 0
    for i in range(M):
        res_int += res(q, p[i], 1)
    return np.linalg.inv(res_int/M)

# Calculates the asymptotic covariance
def cov(p_K5, q_K5, K):
    M = len(p_K5)
    v = res_alle(M, q_K5, p_K5)
    
    cov_X5_X = -np.sum(v, axis=0)
    var_X5 = np.sum(v)
    C_extended = np.zeros((K, K))
    C_extended[:(K-1), :(K-1)] = v    
    C_extended[(K-1), :(K-1)] = cov_X5_X
    C_extended[:(K-1), (K-1)] = cov_X5_X    
    C_extended[(K-1), (K-1)] = var_X5  
    
    return C_extended

def calc_v(M, q, p):
    res_int = 0
    for i in range(M):
        res_int += res(q, p[i], 0)
    return res_int/M

def objective_function(lambda_vec, X_real, Sigma, Sigma_0):
    lambda_vec = np.array(lambda_vec)
    diff = lambda_vec -  X_real

    return np.dot(diff.T, np.dot(Sigma_0, diff))

# Calculate the asymtotic distribution of one individual

def calc_distribution(hat_q, hat_p, K, n_simulations, M):
    Sigma = cov(hat_p, hat_q, K)
    Sigma_1 = calc_v(M,hat_q, hat_p)
    mu = [0]*K
    if(min(hat_q)> 10**(-3)): # Normal Distribution
        return Sigma
    else: # boundary
        max_values2 = []
        for _ in range(n_simulations):
            
            indices_leq_0 =np.where(hat_q < 10**(-3))[0]
            indices_geq_0 =np.where(hat_q > 1-10**(-3))[0]

            X_real = np.random.multivariate_normal(mu, Sigma)

            bounds = [(-100, 100)] * len(mu)  
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w)}]

            for i in indices_leq_0:
                bounds[i] = (0, 100) 
            for i in indices_geq_0:
                    bounds[i] = (-100, 0)
            result = minimize(objective_function, np.zeros_like(mu), args=(X_real,Sigma,Sigma_1, ),  constraints = cons, bounds =bounds)
            max_values2.append(result.x.tolist())
            
    return max_values2, Sigma

index_ind = 0 # Index of the individual that should be considered

distr = calc_distribution(np.array(df_q.iloc[index_ind,:]), df_p.to_numpy(), K, 10000, 55)

#%%
# Plot the results
ind = 0 # index of the population that should be considered
plt.figure(figsize=(12, 6))
plt.hist(np.array(distr[0])[:, 0], bins=32, color='blue', alpha=1, edgecolor='black', density=True)  # Normierung aktiviert
plt.xlabel(rf'$\hat \lambda_{{{ind+1}}}$', fontsize  = 24)
plt.ylabel('Density', fontsize = 24)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.show()