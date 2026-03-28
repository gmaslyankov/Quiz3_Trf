import numpy as np
from scipy.stats import norm
import math

def credit_spread_Merton(v, M, T, sigma, r, theta=1.0):
    d1 = (math.log(v/M) + (r + (sigma**2)/2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    
    # D = M*math.exp(-r*T) - opt_price('put', v, M, sigma, T, r)
    D = M*math.exp(-r*T)*norm.cdf(d2) + v*theta*norm.cdf(-d1)

    cs = -(1/T)*math.log(D/M) - r
    return cs*10000 # in basis points

def credit_spread_Merton_jump(v, M, T, sigma, r, lambd, mu_J, sigma_J, max_jumps=50):
    equity = 0.0
    for n in range(max_jumps):
        prob = ((lambd*(1 + mu_J)*T)**n)*math.exp(-lambd*(1 + mu_J)*T)/math.factorial(n)
        sigma_nod = math.sqrt(sigma**2 + (n*(sigma_J**2))/T)
        r_nod = r - lambd*mu_J + (n*math.log(1 + mu_J))/T
        
        equity += prob*opt_price('call', v, M, sigma_nod, T, r_nod)
    
    D = v - equity
    cs = -(1/T)*math.log(D/M) - r
    return cs*10000 # in basis points

def credit_spread_callable_debt(v, M, T, sigma, r, rho, n=500, theta=1.0):
    delta_t = T/n
    u = math.exp(sigma*math.sqrt(delta_t))
    d = math.exp(-sigma*math.sqrt(delta_t))
    
    pi = (math.exp(r*delta_t) - d)/(u-d) # EMM probability of up move

    tree = np.zeros((n, n))

    # fill preultimate column with zero coupon debt value from Merton model
    for j in range(n):
        v_j = v*(u**j)*(d**(n- 1 - j)) # value of debt at that position
        K = M*math.exp(-rho*(T - (n-1)*delta_t)) # redemption value
        tree[j, n-1] = min(credit_spread_Merton(v_j, M, delta_t, sigma, r, theta), K)

    # compute debt value at other periods going backward
    for t in range(n-2, -1, -1):
        K = M*math.exp(-rho*(T - t*delta_t))
        for j in range(t+1):
            tree[t, j] = min(K, (pi*tree[t+1, j+1] + (1-pi)*tree[t+1, j])*math.exp(-r*delta_t))
    
    print(tree)
    D = tree[0, 0]
    cs = -(1/T)*math.log(D/M) - r
    return cs*10000 # in basis points

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
    
