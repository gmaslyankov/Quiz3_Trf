import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal as multi_norm

import math

def credit_spread_Merton(v, M, T, sigma, r, theta=1.0, verbous = False):
    d1 = (math.log(v/M) + (r + (sigma**2)/2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    
    # D = M*math.exp(-r*T) - opt_price('put', v, M, sigma, T, r)
    D = M*math.exp(-r*T)*norm.cdf(d2) + v*theta*norm.cdf(-d1)

    cs = -(1/T)*math.log(D/M) - r

    if verbous:
        print(f'Debt value = {D}')
        print(f'credit spread = {cs*10000}')

    return cs*10000, D # spread in basis points

def credit_spread_Merton_jump(v, M, T, sigma, r, lambd, mu_J, sigma_J, max_jumps=50, verbous = False):
    equity = 0.0
    for n in range(max_jumps):
        prob = ((lambd*(1 + mu_J)*T)**n)*math.exp(-lambd*(1 + mu_J)*T)/math.factorial(n)
        sigma_nod = math.sqrt(sigma**2 + (n*(sigma_J**2))/T)
        r_nod = r - lambd*mu_J + (n*math.log(1 + mu_J))/T
        
        equity += prob*opt_price('call', v, M, sigma_nod, T, r_nod)
    
    D = v - equity
    cs = -(1/T)*math.log(D/M) - r

    if verbous:
        print(f'Debt value = {D}')
        print(f'credit spread = {cs*10000}')

    return cs*10000, D # in basis points

def credit_spread_callable_debt(v, M, T, sigma, r, rho, n=500, theta=1.0, verbous = False):
    delta_t = T/n
    u = math.exp(sigma*math.sqrt(delta_t))
    d = math.exp(-sigma*math.sqrt(delta_t))
    
    pi = (math.exp(r*delta_t) - d)/(u-d) # EMM probability of up move

    tree = np.zeros((n, n))

    # fill preultimate column with zero coupon debt value from Merton model
    for j in range(n):
        # print(f'j = {j}')
        v_j = v*(u**j)*(d**(n- 1 - j)) # value of debt at that position
        # print(f'v_j = {v_j}')

        K = M*math.exp(-rho*(T - (n-1)*delta_t)) # redemption value
        # print(f'K = {K}')
        tree[j, n-1] = min(credit_spread_Merton(v_j, M, delta_t, sigma, r, theta)[1], K)
        # print(f'cell val = {tree[j, n-1]}')

    # compute debt value at other periods going backward
    for t in range(n-2, -1, -1):
        # print(f't = {t}')
        K = M*math.exp(-rho*(T - t*delta_t))
        # print(f'K = {K}')

        for j in range(t+1):
            # print(f'j = {j}')

            tree[j, t] = min(K, (pi*tree[j+1, t+1] + (1-pi)*tree[j, t+1])*math.exp(-r*delta_t))
            # print(f'cell val = {tree[j, t]}')

    # print(tree)
    D = tree[0, 0]
    cs = -(1/T)*math.log(D/M) - r

    if verbous:
        print(f'Debt value = {D}')
        print(f'credit spread = {cs*10000}')

    return cs*10000, D # in basis points

def credit_spread_convertible_debt(v, M, T, sigma, r, q, n=500, theta=1.0, verbous = False):
    delta_t = T/n
    u = math.exp(sigma*math.sqrt(delta_t))
    d = math.exp(-sigma*math.sqrt(delta_t))
    
    pi = (math.exp(r*delta_t) - d)/(u-d) # EMM probability of up move

    tree = np.zeros((n, n))

    # fill preultimate column with zero coupon debt value from Merton model
    for j in range(n):
        # print(f'j = {j}')
        v_j = v*(u**j)*(d**(n- 1 - j)) # value of debt at that position
        # print(f'v_j = {v_j}')

        # print(f'K = {K}')
        tree[j, n-1] = max(q*v_j, convertible_bond_value(v_j, M, sigma, delta_t, r, q, theta))
        # print(f'cell val = {tree[j, n-1]}')

    # compute debt value at other periods going backward
    for t in range(n-2, -1, -1):
        # print(f't = {t}')

        for j in range(t+1):
            # print(f'j = {j}')
            v_j = v*(u**j)*(d**(t - j)) # value of debt at that position
            tree[j, t] = max(q*v_j, (pi*tree[j+1, t+1] + (1-pi)*tree[j, t+1])*math.exp(-r*delta_t))
            # print(f'cell val = {tree[j, t]}')

    # print(tree)
    D = tree[0, 0]
    cs = -(1/T)*math.log(D/M) - r

    if verbous:
        print(f'Debt value = {D}')
        print(f'credit spread = {cs*10000}')
        
    return cs*10000, D # in basis points

def credit_spread_callable_convertible_debt(v, M, T, sigma, r, rho, q, n=500, theta=1.0, verbous = False):
    delta_t = T/n
    u = math.exp(sigma*math.sqrt(delta_t))
    d = math.exp(-sigma*math.sqrt(delta_t))
    
    pi = (math.exp(r*delta_t) - d)/(u-d) # EMM probability of up move

    tree = np.zeros((n, n))

    # fill preultimate column with zero coupon debt value from Merton model
    for j in range(n):
        # print(f'j = {j}')
        v_j = v*(u**j)*(d**(n- 1 - j)) # value of debt at that position
        # print(f'v_j = {v_j}')

        K = M*math.exp(-rho*(T - (n-1)*delta_t)) # redemption value
        # print(f'K = {K}')
        tree[j, n-1] = min(max(q*v_j, convertible_bond_value(v_j, M, sigma, delta_t, r, q, theta)), max(K, q*v_j))
        # print(f'cell val = {tree[j, n-1]}')

    # compute debt value at other periods going backward
    for t in range(n-2, -1, -1):
        # print(f't = {t}')
        K = M*math.exp(-rho*(T - t*delta_t))
        # print(f'K = {K}')

        for j in range(t+1):
            # print(f'j = {j}')
            v_j = v*(u**j)*(d**(t - j)) # value of debt at that position
            tree[j, t] = min(max(q*v_j, (pi*tree[j+1, t+1] + (1-pi)*tree[j, t+1])*math.exp(-r*delta_t)), max(q*v_j, K))
            # print(f'cell val = {tree[j, t]}')

    # print(tree)
    D = tree[0, 0]
    cs = -(1/T)*math.log(D/M) - r

    if verbous:
        print(f'Debt value = {D}')
        print(f'credit spread = {cs*10000}')
        
    return cs*10000, D # in basis points

def credit_spread_short_long_term_debt(v, M, m, T, t, sigma, r, theta=1.0, verbous = False):
    z1 = (math.log(v/m) + (r + (sigma**2)/2)*t)/(sigma*math.sqrt(t))
    z2 = z1 - sigma*math.sqrt(t)

    z3 = (math.log(v/M) + (r + (sigma**2)/2)*T)/(sigma*math.sqrt(T))
    z4 = z3 - sigma*math.sqrt(T)

    var_cov1 = np.zeros((2,2))
    var_cov1[0,0] = var_cov1[1,1] = 1.0
    var_cov1[0,1] = var_cov1[1,0] = math.sqrt(t)/math.sqrt(T)

    var_cov2 = np.zeros((2,2))
    var_cov2[0,0] = var_cov2[1,1] = 1.0
    var_cov2[0,1] = var_cov2[1,0] = -math.sqrt(t)/math.sqrt(T)
    
    d = m*math.exp(-r*t)*norm.cdf(z2) + (m/(m+M))*theta*v*norm.cdf(-z1)
    D = M*math.exp(-r*T)*multi_norm.cdf(np.array([z2, z4]), None, var_cov1) + theta*v*multi_norm.cdf(np.array([z1, -z3]), None, var_cov2) + (M/(m+M))*theta*v*norm.cdf(-z1)
    
    cs_long = -(1/T)*math.log(D/M) - r
    cs_short = -(1/t)*math.log(d/M) - r

    if verbous:
        print(f'Long term Debt value = {D}')
        print(f'credit spread = {cs_long*10000}')
        print()
        print(f'Short term Debt value = {d}')
        print(f'credit spread = {cs_short*10000}')
        
    return cs_long*10000, D, cs_short*10000, d # credit spreads in basis points

####################
# Helper functions #
####################

def opt_price(type, S, K, sigma, T, r):
    d1 = (math.log(S/K) + (r + (sigma**2)/2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)

    call_price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)

    if type == 'call':
        return call_price
    elif type == 'put':
        return call_price - S + K*math.exp(-r*T)
    else:
        raise ValueError(f'Incorrect option type |{type}| was given')
    
def convertible_bond_value(v, M, sigma, T, r, q, theta=1.0):
    d1 = (math.log(v/M) + (r + (sigma**2)/2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)

    d3 = (math.log(q*v/M) + (r + (sigma**2)/2)*T)/(sigma*math.sqrt(T))
    d4 = d3 - sigma*math.sqrt(T)

    return v*theta*norm.cdf(-d1) + M*math.exp(-r*T)*(norm.cdf(d2) - norm.cdf(d4)) + q*v*norm.cdf(d3)
