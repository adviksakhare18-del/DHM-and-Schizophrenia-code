# 10. THIS FILE USES fit_potential() ON THE TEST DATASET TO GET A NEW CSV OF FITTED PARAMETERS FOR THE TEST DATASET
from useful import *
import mpmath as m
import csv
from Claude_curve_fit import fit_potential
from Series_plot import buildarray

def main():
    rawmatrix = []
    with open('labeled_action_potentials.csv','r') as test1:
        test = csv.reader(test1)
        for row in test:
            rawmatrix.append(row)
    
    rawmatrix = rawmatrix[1:]
    matrix = []
    results = []
    for row in rawmatrix: # REMOVE THE LIMITER AND FINISH PROCESSING EVERYTHING
        matrix.append(multifunc(row[1:],float)) # Contains time series of patients turned into values
        results.append(row[0]) # Contains statuses of patients
    
    time = linspace(0,80,0.05)
    test_data_params = [] # Finds parameter values
    i = 0
    for row in matrix:
        params = fit_potential(time,row) # Times and then values
        i += 1
        print(f"Finished: {i}") # Progress tracker
        test_data_params.append(params)
    
    with open('test_param_values.csv','a',newline = "") as testp1:
        testp = csv.writer(testp1)
        testp.writerow(["mg",'b','j','k','l','base','health','decay'])
        ind = 0
        for row in test_data_params:
            # Decay = -k/2j
            add_arr = [results[ind],-row[3]/(2*row[2])]
            row = arraymerger([row,add_arr])
            ind += 1
            testp.writerow(row)
    
    print("FINISHED ALL ROWS")
    

"""
test data in the form
type - values for a person
"""

if __name__ == "__main__":
    main()