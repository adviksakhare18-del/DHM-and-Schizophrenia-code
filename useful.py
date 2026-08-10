# THIS IS THE USEFUL.PY FILE WITH ALL MY FUNCTIONS TO MAKE THE CODE CLEANER
import matplotlib.pyplot
import random
import mpmath
import numpy #Unused
import scipy #Unused
import pandas #Unused

"""
This is a file with functions I have written to streamline my work

Function Contents:

- Linspace: An evenly distributed array of floats between the bounds, differing by interval
- Quickplot: A function that takes in 2 arrays and 3 strings and outputs a fully formatted graph
- Quickscatter: A function that does quickplot, but with scatterplots instead
- Multifunc: A function that takes in an array and outputs an array of individual functions
- Multifuncdbl: A function that takes 2 arrays and a multivar function and outputs an array of arrays [f(x,y1) to f(x,yn)]
- Randarray: Generates array of uniformly distributed numbers in range
- Vecdot: Outputs the dot product of two equal length vectors
- Vecoperation: Adds/subtracts vectors of equal length
- Veccross: Finds cross prod from 1st 3 entries of 2 array vectors
- Strtointarr: Converts a string array to a float array 
- Vecproj: Returns the projection of arr1 onto arr2 as an array
- Vecmag: Returns the magnitude of a vector as a float
- Vecscale: Scales a vector array by a scalar, can be used to scale function outputs too.
- Mattranspose: Takes in a matrix or array of arrays and outputs its transpose. Used to turn function rows into columns.
- Invsncdf: Takes in a number and outputs the number that would output the first number if it was a norm.s.cdf
- Differentiate: Takes two arrays and differentiates one with respect to the other
- Arraymerger: Merges all things from an array of arrays into one big array
- Integrate: WIP

TO IMPORT ALL, DO FROM USEFUL.PY IMPORT *
"""

def main1():
    start = float(input("Start: "))
    stop  = float(input("End: "))
    step = float(input("Step: "))
    xaxis = linspace(start, stop, step)
    print(xaxis)
    arr1 = linspace(0,20,2)

#-------------------------------------------------------------------------
"""
This is the linspace function
It takes in 2 float values (minimum) for the start and stop
An optional one for th step is also there (default = 1)
Errors with input are upto the user to handle, as they may want to handle it differently
to what I would do.
Takes in 2-3 floats, outputs an array of floats until equal to or less than the stop value
"""

def linspace(start, stop, step=1):
    output = []
    output.append(start)
    while True:
        start += step
        if start <= stop:
            output.append(start)
            continue
        else:
            return output

#--------------------------------------------------------------------------
"""
This is a function that takes in the x and y arrays, the title and label strings
It outputs a single function graph using a normal .plot() command.
Make sure to call mpl.show() and mpl.legend()

"""
def quickplot(xarr, yarr, title='', xaxis='', yaxis='',label=None):
    matplotlib.pyplot.plot(xarr,yarr,label=label)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.xlabel(xaxis)
    matplotlib.pyplot.ylabel(yaxis)
    matplotlib.pyplot.axvline(x=0,color='black',linewidth=1)
    matplotlib.pyplot.axhline(y=0,color='black',linewidth=1)
    matplotlib.pyplot.grid(True, axis='both')
    if label:
        matplotlib.pyplot.legend()

#---------------------------------------------------------------------------
"""
This is a function that takes in the x and y arrays, the title and label strings
It outputs a single function graph using a .scatter() command instead of .plot()
You need to call .show() and maybe .legend() after plotting.
"""
def quickscatter(xarr, yarr, title='', xaxis='', yaxis='',label=None):
    matplotlib.pyplot.scatter(xarr,yarr,label=label)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.xlabel(xaxis)
    matplotlib.pyplot.ylabel(yaxis)
    matplotlib.pyplot.axvline(x=0,color='black',linewidth=1)
    matplotlib.pyplot.axhline(y=0,color='black',linewidth=1)
    matplotlib.pyplot.grid(True, axis='both')
    if label:
        matplotlib.pyplot.legend()
#-------------------------------------------------------------------------

"""
This is a function that takes in the x array, applies function to each entry individually
and outputs an array of the same length with the outputs of the function 
Takes in and outputs integer/float values
"""
def multifunc(xarr, function): 
    length = len(xarr)
    output = []
    for i in range(length):
        add = function(xarr[i])
        output.append(add)
    return output
#-----------------------------------------------------------------------
"""
This is a function like multifunc, except here it takes a function and two arrays for two
variables x and y, then outputs an array of arrays of the form [f(x,y1),f(x,y2)...f(x,yn)]
Needs an unpacking variable
"""
def multifuncdbl(xarr, yarr, IIvarfunction):
    output = []
    for i in yarr:
        arradd = []
        for j in xarr:
            arradd.append(IIvarfunction(j,i))
        output.append(arradd)
    return output

#Example of usage:
"""
output = multifuncdbl(x,k,function)
for ykarr in output:
    quickplot(x,ykarr)
mpl.legend()
mpl.show()
"""
#-----------------------------------------------------------------------
"""
This is a function that takes 2 floats, then an int, and generates the number
of randomly distributed points across the interval
"""
def randarray(min, max, number):
    output = []
    for _ in range(number):
        add = random.uniform(min,max)
        output.append(add)
    return output
#-----------------------------------------------------------------------
"""
This is a function that takes in 2 arrays of numbers (eql length) as vectors 
and outputs the dot product of them both as a number
"""
def vecdot(arr1,arr2):
    dotinter = []
    for i in range(max(len(arr1),len(arr2))):
        add = arr1[i]*arr2[i]
        dotinter.append(add)
    return sum(dotinter)
#-----------------------------------------------------------------------
"""
This is a function that finds the sum or difference of two vectors.
It either adds or subtracts two same length arrays, deciding using another string.
It returns an array of the same length 
"""

def vecoperation(arr1,arr2,plusorminus):
    opinter = []
    op = plusorminus
    if op == "+":
        for i in range(max(len(arr1),len(arr2))):
            add = arr1[i]+arr2[i]
            opinter.append(add)
        return opinter
    elif op == '-':
        for i in range(max(len(arr1),len(arr2))):
            diff = arr1[i]-arr2[i]
            opinter.append(diff)
        return opinter
    else:
        print("Not working")
#-----------------------------------------------------------------------
"""
This is a function that takes in three arrays(len>=3) of floats as vectors
and outputs their cross product as a vector
"""
def veccross(arr1,arr2):
    cross = [
        arr1[1]*arr2[2]-arr1[2]*arr2[1],
        arr1[2]*arr2[0]-arr1[0]*arr2[2],
        arr1[0]*arr2[1]-arr1[1]*arr2[0]
        ]
    return cross
#-----------------------------------------------------------------------
"""
This function takes in an array of stringform numbers and outputs them as 
float values to clean up input and outputs an array of clean numbers
"""
def strtointarr(arr):
    output = []
    for i in range(len(arr)):
        output.append(float(arr[i]))
    return output
    ...
#-----------------------------------------------------------------------
"""
This function is used to find the projection of the first vector onto the
second vector and outputs an array
"""
def vecproj(arr1,arr2):
    from useful import vecmag,vecdot,vecscale
    scalar = vecdot(arr1,arr2)/vecmag(arr2)**2
    arrint = vecscale(arr2,scalar)
    output = []
    for i in range(len(arrint)):
        output.append(float(arrint[i]))
    return output
#-----------------------------------------------------------------------
"""
This function takes in a vector in the form of an array and outputs its
pythagorean magnitude as a float
"""
def vecmag(arr):
    from useful import vecdot
    mag = vecdot(arr,arr)
    return mpmath.sqrt(mag)
#-----------------------------------------------------------------------
"""
This function takes in a vector in array form and a scalar, and then it
scales all entries in the array by that number. Can also be used to scale
functions outputs by that amount.
"""
def vecscale(arr,num):
    output = []
    for i in range(len(arr)):
        output.append(num*arr[i])
    return output
#-----------------------------------------------------------------------
"""
This function takes in an array of arrays (matrix) and returns its transpose 
as an array of arrays. an m*n matrix gets turned into an n*m matrix.
"""
def mattranspose(matrix):
    output = []
    for i in range(0,len(matrix[0])):
        row = []
        for j in range(0,len(matrix)):
            row.append(matrix[j][i])
        output.append(row)
    return output
#-----------------------------------------------------------------------
"""
Takes in a number between 0 and 1 and outputs the number x so that PHI(x)
is equal to the inverse of the standard normal cdf.
"""
def invsncdf(x):
    return mpmath.sqrt(2) * mpmath.erfinv(2*x-1)
#----------------------------------------------------------------------
"""
This function accepts two equal length arrays of floats, and outputs an
array with each points' gradients attached to it.
"""
def differentiate(xarr,yarr):
    output = []
    while True:
        for i in range(len(xarr)):
            try:
                dy = (yarr[i+1]-yarr[i])/(xarr[i+1]-xarr[i])
            except IndexError:
                dy = output[i-1] + (output[i-1]-output[i-2])
            except ZeroDivisionError:
                dy = output[i-1] + (output[i-1]-output[i-2])
            output.append(dy)
        return output
#-----------------------------------------------------------------------
"""
This function takes in an array of arrays (like a matrix) and condenses
all the information there into one singular array, in thr order of 
appearance in the initial array.
"""
def arraymerger(arrofarrs):
    output = []
    for array in arrofarrs:
        for entry in array:
            output.append(entry)
    return output

#-----------------------------------------------------------------------
"""
This function inputs two arrays and 2 floats and integrates over the array which it is given.
It passes through the point which it is given, and if not given, it passes through the origin.
NOTE: WIP
"""
def integrate(xarr,yarr, xpt=0, ypt=0):    
    output = []
    position = 0
    while True: # Position finder
        try:
            if xpt > xarr[position]:
                position += 1
            else:
                print(f"Position: after the {position}")
                break
        except IndexError:
            position = len(xarr)
            print(f"Position: after the {position}")
            break
    # The goal is to find position and work backward from it then rebuild the array with the intial point forced into it
    while True:
        backward = []
        Toggle = True
        for i in range(len(xarr)): # Adding xpt to xarray
            if xarr[i] > xpt and Toggle:
                backward.append(xpt)
                Toggle = False
                backward.append(xarr[i])
            else:
                backward.append(xarr[i])
        break
    print(backward)
    
    afterypt = [ypt]
    area = 0
    for i in range(position,len(xarr)):
        area = area + backward[i]
        afterypt.append(yarr[i])
    
    print(f"This is afterypt {xarr}")
                    # Think along the line of prev entry + deltax/2*(oldy+newy)
                    # If going backward, it is prev entry - deltax/2*(oldy+newy)
#-----------------------------------------------------------------------
if __name__ == "__main__":
    main1()