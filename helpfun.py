def ewma(v1, v2, beta): 
    """exponential weighted moving average / lerp"""
    """torch.lerp(v2, v1, beta) other way around!"""    
    return beta*v1 + (1-beta)*v2

def mom_db(avg,yi,beta,i):
    if avg is None: avg=yi
    res = ewma(avg,yi,beta)
    return res, res/(1-beta**(i+1))