# 5. THIS IS THE FILE WHERE WE GENERATE THE 'PARABOLAS OF SEPARATION' FOR 2 PROMISING GRAPHS (SEE PAPER FOR MORE INFO)
# This is where Tests A and B emerged from.
from useful import *
import mpmath as m
import matplotlib.pyplot as mpl
import numpy as np
from Series_plot import buildarray

def main():
    var1 = 'mg'
    var2 = 'decay'
    x1 = np.array(buildarray(var1, 'healthy'), dtype=float)
    y1 = np.array(buildarray(var2, 'healthy'), dtype=float)
    x2 = np.array(buildarray(var1, 'schizophrenic'), dtype=float)
    y2 = np.array(buildarray(var2, 'schizophrenic'), dtype=float)
    x = arraymerger([x1,x2])
    healthycoeff = np.polyfit(x1,y1,deg=2)
    schizocoeff = np.polyfit(x2,y2,deg=2) # Uses linreg to find coeffs of schizo and healthy lines
    avgarray = averagearray(schizocoeff, healthycoeff)
    print(avgarray)
    # Finds the linreg values of a single line
    schizofunc = np.poly1d(schizocoeff)
    healthyfunc = np.poly1d(healthycoeff)
    avgfunc = np.poly1d(avgarray)
    
    quickscatter(x1,y1, label = "Healthy")
    quickscatter(x2,y2, label = "Schizophrenic")
    x = linspace(0,max(x),max(x)/1000)
    poly = multifunc(x,avgfunc)
    quickplot(x,poly,f'{var1} vs {var2}',var1, var2, label=f"{avgarray[0]}{var1}^2 + {avgarray[1]}{var1} + {avgarray[2]}")
    mpl.show()

def averagearray(arr1, arr2):
    output = []
    for i in range(0,len(arr1)):
        output.append((arr1[i]+arr2[i])/2)
    return np.array(output)


if __name__ == "__main__":
    main()