# 7. THIS FILE PARSES param_values_final.csv (THE REAL DATASET) FOR TYPE 1 AND TYPE 2 ERRORS AND COUNTS THEM BOTH.
from Series_plot import buildarray
from useful import *
import mpmath as m

# Here we can consider only test A, as it can work by itself to diagnose 
# schizophrenia to a good degree

def main():
    mghealthy = multifunc(buildarray('mg','healthy'),float)
    mgschizo = multifunc(buildarray('mg','schizophrenic'),float)
    mg = arraymerger([mghealthy,mgschizo])
    
    stathealthy = buildarray('health','healthy')
    statschizo = buildarray('health','schizophrenic')
    stat = arraymerger([stathealthy, statschizo])
    
    decayhealthy = multifunc(buildarray('decay','healthy'),float)
    decayschizo = multifunc(buildarray('decay','schizophrenic'),float)
    decay = arraymerger([decayhealthy,decayschizo])
    
    t1,t2 = errors(mg,decay,stat)
    print(f"No of type 1 errors (false +): {t1}")
    print(f"No of type 2 errors (false -): {t2}")
    # Use this to find P(type 1 error) and P(type 2 error)

def errors(mg,decay,stat):
    t1,t2 = 0,0
    for i in range(0,len(mg)):
        A = False # This is the variable that confirms if you have it or not.
        if decay[i] > (2.47608430e-07)*mg[i]**2 + (3.59130201e-03)*mg[i] + (1.11493342e-01):
            A = True # Make sure to use |decay| in equation
        if A and stat[i] == 'healthy': # False positive (type 1)
            t1 += 1
        elif not A and stat[i] == 'schizophrenic': # False negative (type 2)
            t2 += 1
    
    return t1,t2

if __name__ == "__main__":
    main()
