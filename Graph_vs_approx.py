# EXTRA: THIS IS THE FILE THAT I USED TO VISUALISE THE APPROXIMATION GRAPH VS THE ACTUAL AVG ACTION POTENTIAL DATA OF A COHORT WITH STDDEV VARIATIONS SHOWN
from useful import *
import matplotlib.pyplot as mpl
import mpmath as m
import numpy as np
import pandas as pd
import csv
import numpy as np
from typing import Optional
from scipy.optimize import minimize, OptimizeResult
from Claude_curve_fit import *

def main():
    with open('healthy_action_potential.csv','r') as file1:
        file = csv.reader(file1)
        rawmatrix = []
        for row in file:
            rawmatrix.append(row)
        yavg,ystd = outarray(rawmatrix[1:])
        yup = yhigh(yavg,ystd)
        ydown = ylow(yavg,ystd)
        x1 = linspace(0,80,0.05)
        y1 = multifunc(x1,potential)
        quickplot(x1,yavg, label = "Actual data")
        quickplot(x1,yup,label = "Upper bound")
        quickplot(x1,ydown, label = "Lower bound")
        quickplot(x1,y1,"Action potential of a healthy human","Time (ms)","Action Potential (mV)",label="Approximation")
        mpl.show()

def yhigh(yavg,ystd):
    output = []
    for i in range(0,len(yavg)):
        output.append((yavg[i]+ystd[i]))
    return output

def ylow(yavg,ystd):
    output = []
    for i in range(0,len(yavg)):
        output.append((yavg[i]-ystd[i]))
    return output

def outarray(matrix):
    yavg = []
    ystd = []
    for row in matrix:
        try:
            rowext = multifunc(row[1:], float) #Change it back to 1:
            yavg.append(rowext[-2])
            ystd.append(rowext[-1])
        except TypeError: # This is NECESSARY please dont remove it
            print("PASSED")
            pass
    return yavg,ystd

def avg(arr):
    return sum(arr)/len(arr)

def inter(x, int):
    return round(x,int)

# Solved for ODE jv''+kv'+lv=0 using laplace transform (see paper)
def potential(t):
    # --- Parameters ---
    mg = 402.2442 # linear increase gradient
    b = 0.1662 # y intercept of linear increase
    j = 0.0261 # strength of restorative force (equivalent to spring constant in DHM system)
    k = 0.0771 # 0 < k < sqrt(4jl), or else you'll get complex solutions or cause the resistive term to add to jv''
    l = 0.0570 # constant of proportionality, keeps stuff clean
    base = -67.85 # Baseline potential in mV
    shift = b/mg + 5 # Ability to shift along time axis (controls point of start for linear section)
    # ------------------
    
    Q = m.sqrt((4*j*l-k**2)/(4*j**2)) # Equivalent to big K from book
    C = mg + b*k/(2*j)
    t -= shift # Shifts graph to right by shift value
    
    # --- Piecewise Function ---
    if t<-b/mg: # Baseline
        y = base
    elif -b/mg <= t < 0: # Depolarisation 
        y = mg*t+b+base
    elif t>=0: # Repolarisation, Hyperpolarisation and return to baseline value
        y = m.exp(-k*t/(2*j))* (b*m.cos(Q*t) + C/Q*m.sin(Q*t))+base
    #---------------------------
    return y

if __name__ == "__main__":
    main()