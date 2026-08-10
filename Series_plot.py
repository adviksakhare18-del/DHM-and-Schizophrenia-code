# 4. THIS IS A FILE THAT WILL GENERATE A QUICK SCATTERPLOT OF TWO VARIABLES OF YOUR CHOICE. GREAT FOR PATTERN VISUALISATION.
from useful import *
import mpmath as m
import matplotlib.pyplot as mpl
import csv

# Do Probability anaylses and try to analyse the conditional probability of schizophrenia

def main():
    var1 = input("X axis: ")
    var2 = input("Y axis: ")
    x1 = multifunc(buildarray(var1,"healthy"),float)
    x2 = multifunc(buildarray(var1,'schizophrenic'),float)
    y1 = multifunc(buildarray(var2,'healthy'),float)
    y2 = multifunc(buildarray(var2,'schizophrenic'),float)
    
    x = arraymerger([x1,x2])
    y = arraymerger([y1,y2]) 
    # quickscatter(x,y) # Collective Graph Generator (both groups)
    
    quickscatter(x1,y1,f"Graph of {var1} VS. {var2}",f"{var1}",f"{var2}", label='Healthy') # Healthy group plots
    quickscatter(x2,y2,f"Graph of {var1} VS. {var2}",f"{var1}",f"{var2}", label = "Schizophrenic") # Schizo group plots
    mpl.legend()
    mpl.show()

def buildarray(var,statstr): # Add the variable you are looking for and the state of the person and get the values of it from everyone
    keydict = {
    "mg":0,
    "b":1,
    "j":2,
    "k":3,
    "l":4,
    "base":5,
    "health":6, # Not a float
    "decay":7,
    "frequency":8}
    # Use healthy or schizophrenic for statstr
    output = []
    rawinput = []
    with open("param_values_final.csv",'r') as paramvalues:
        values = csv.reader(paramvalues)
        for row in values:
            rawinput.append(row)
    matrix = rawinput[1:] # Truncated top row of matrix
    num = keydict[var]
    for row in matrix:
        if row[6] == statstr:
            output.append(row[num])
    return output


if __name__ == "__main__":
    main()