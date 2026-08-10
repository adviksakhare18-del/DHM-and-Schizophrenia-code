# 8. THIS IS THE SCRIPT THAT WILL LET YOU TEST YOUR OWN ARRAY OF POTENTIAL DATAPOINTS, FITTING IT AND PROVIDING A DIAGNOSIS 
from numpy.matlib import False_
from useful import *
from Claude_curve_fit import fit_potential
import csv
import numpy as np
import matplotlib.pyplot as mpl
import mpmath as m

def main():
    rawmatrix = []
    with open('schizophrenia_action_potential.csv','r') as schizo1: # Can chane file b/t healthy and schizophrenia
        schizo = csv.reader(schizo1)
        for row in schizo:
            rawmatrix.append(row)
    tmatrix = mattranspose(rawmatrix[1:])
    randseries = np.array(tmatrix[1:], dtype=float)[5] # Can change this number here too
    
    """
    NOTE: You are free to replace the value of y below with whatever array of floats
    you choose, and it will calculate the parameters and give you your diagnosis with
    that particular set.
    Output = [mg, b, j, k, l, base]
    decay = -k/2j
    """
    x = linspace(0,80,0.05)
    y = randseries
    param = fit_potential(x,y)
    """
    for y in randseries:
        param = fit_potential(x,y)
        print(param)
        mg = param[0]
        j = param[2]
        decay = -param[3]/(2*param[2]) # decay = -k/2j
        absdecay = -decay
        result = check(mg,j,absdecay) # try decay and absdecay
        print(f"The probability of you having Schizophrenia is: {round(result*100,3)} %")
    """
    mg = param[0]
    j = param[2]
    decay = -param[3]/(2*param[2]) # decay = -k/2j
    absdecay = -decay
    result = check(mg,j,absdecay)
    print(f"The probability of you having Schizophrenia is: {round(result*100,3)} %")
    ...

def check(mg,j,decay):
    # mg/decay : (2.47608430e-07)x^2 + (3.59130201e-03)x + (1.11493342e-01) up +
    # mg/j : (8.07675322e-08)x^2 + (-1.43973630e-04)x + (7.00188346e-02) down +
    A,B = False, False
    
    if decay > (2.47608430e-07)*mg**2 + (3.59130201e-03)*mg + (1.11493342e-01):
        A = True
    
    if j < (8.07675322e-08)*mg**2 + (-1.43973630e-04)*mg + (7.00188346e-02):
        B = True
    # Bayes Adjusted values
    if A and B:
        p = 1.0
    elif A and not B:
        p = 1.0
    elif not A and B:
        p = 0.004659146640510054
    elif not A and not B:
        p = 0.0003864921010676845
    
    return p

if __name__ == "__main__":
    main()
