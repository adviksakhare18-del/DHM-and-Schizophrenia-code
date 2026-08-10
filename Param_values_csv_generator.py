# 2. THIS IS THE FILE THAT USES THE NELDOR-MEAD ALGORITHM TO GENERATE THE CURVE FITTED VALUES FOR EACH PERSON'S DATA
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
    x = linspace(0,80,0.05)
    rawmatrix = []
    filename = "healthy_action_potential.csv" # Can change name of input file here
    with open(filename,'r',newline="") as healthy1:
        healthy = csv.reader(healthy1)
        print("Finished 1") #
        for row in healthy:
            rawmatrix.append(row[1:-2])
        print("Finished 2") #
    matrix = mattranspose(rawmatrix[1:]) # Rows represent individual y values
    finalmatrix = []
    if filename == "healthy_action_potential.csv":
        result = "healthy"
    elif filename == 'schizophrenia_action_potential.csv':
        result = "schizophrenic"
    else:
        result = "Null"
    
    i = 0
    for series in matrix: # Remove selection
        output = fit_potential(x,series)
        finalmatrix.append(np.append(output,result))
        i += 1
        print(i)
    print("Finished 3") # 
    
    # Doing same thing for schizo now
    rawmatrix = []
    filename = "schizophrenia_action_potential.csv" # Can change name of input file here
    with open(filename,'r',newline="") as schizo1:
        schizo = csv.reader(schizo1)
        print("Finished 4") #
        for row in schizo:
            rawmatrix.append(row[1:-2])
        print("Finished 5") #
    matrix = mattranspose(rawmatrix[1:]) # Rows represent individual y values
    if filename == "healthy_action_potential.csv":
        result = "healthy"
    elif filename == 'schizophrenia_action_potential.csv':
        result = "schizophrenic"
    else:
        result = "Null"
    for series in matrix: # Remove selection
        output = fit_potential(x,series)
        finalmatrix.append(np.append(output,result))
        i += 1
        print(i)
    print("Finished 3") # 
    
    with open("param_values_both.csv",'a',newline="") as file1:
        file = csv.writer(file1)
        file.writerow(['mg','b','k','l','base','Health'])
        for line in finalmatrix:
            file.writerow(line)
    
    print("Finished 4")

"""
Structure of csv file:
mg|b|j|k|l|base|Health <- Titles
...|...|...|healthy <- Values per person

"""

if __name__ == "__main__":
    main()


