import numpy as np 

def normal_cdf(rv_input, mean=0, std=1):
    from scipy import stats
    result = stats.norm.cdf(rv_input, loc=mean, scale=std)
    return result 

def normal_ppf(quantile, mean=0, std=1):
    from scipy import stats 
    result = stats.norm.ppf(quantile, loc=mean, scale=std)
    return result


def lognormal_cdf(rv_input, stddev, scale=1):
    from scipy import stats
    result = stats.lognorm.cdf(rv_input, stddev, scale=scale)

    return result 

def uniform_cdf(rv_input, low=0, high=1):
    from scipy import stats
    result = stats.uniform.cdf(rv_input, loc=low, scale=high-low)
    return result 

def binom_pmf(n, k, p):
    from scipy import stats 
    result = stats.binom(n=n, p=p).pmf(k=k)
    return result