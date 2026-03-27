from scipy.stats import norm
import math

def credit_spread_Merton(v, M, T, sigma, r):
    D = M*math.exp(-r*T) - opt_price('put', v, M, sigma, T, r)
    cs = -(1/T)*math.log(D/M) - r
    return cs*10000 # in basis points

def credit_spread_Merton_jump(v, M, T, sigma, r, lambd, mu_J, sigma_J):
    equity = 0.0
    for n in range(50):
        prob = ((lambd*(1 + mu_J)*T)**n)*math.exp(-lambd*(1 + mu_J)*T)/math.factorial(n)
        sigma_nod = math.sqrt(sigma**2 + (n*(sigma_J**2))/T)
        r_nod = r + (n*math.log(1 + mu_J))/T
        equity += prob*opt_price('call', v, M, sigma_nod, T, r_nod)
    
    D = v - equity
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
    
