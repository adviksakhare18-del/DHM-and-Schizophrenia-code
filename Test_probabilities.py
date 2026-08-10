# 11. THIS IS THE FILE THAT CALCULATES THE BAYES ADJUSTED ACCURACY PROBABILITIES FOR THE TEST DATASET (FOR PERFORMANCE CONFIRMATION).
from useful import *
import mpmath as m
import csv


def main():
    mghealthy = multifunc(buildarraytest('mg','healthy'),float)
    mgschizo = multifunc(buildarraytest('mg','schizophrenic'),float)
    mgall = arraymerger([mghealthy, mgschizo])
    
    decayhealthy = multifunc(buildarraytest('decay','healthy'),float)
    decayschizo = multifunc(buildarraytest('decay','schizophrenic'),float)
    decayall = arraymerger([decayhealthy, decayschizo])
    
    jhealthy = multifunc(buildarraytest('j','healthy'),float)
    jschizo = multifunc(buildarraytest('j','schizophrenic'),float)
    jall = arraymerger([jhealthy, jschizo])
    
    healthyarr = buildarraytest("health",'healthy')
    schizoarr = buildarraytest('health','schizophrenic')
    actualresult = arraymerger([healthyarr, schizoarr])
    
    data_all1 = [mgall, decayall, jall]
    data_all = mattranspose(data_all1)       # data as individual rows for a person - [mg, decay, j]
    
    resultA = multifunc(data_all, testA)
    resultB = multifunc(data_all, testB) # Find why its all falses for resultA
    
    # Finds raw amount of A correct/wrong and B correct/wrong
    counts = counter(resultA, resultB, actualresult)
    print(counts)
    
    # Find P(schizophrenic|result) and P(correct diagnosis|result) for AB, AB', A'B, A'B'
    probs = joint_result_probs(resultA, resultB, actualresult)
    labels = ["AB", "AB'", "A'B", "A'B'"]
    print("Order:", labels)
    print("P(schizophrenic|result):", probs["p_schizo"])
    print("P(correct diagnosis|result):", probs["p_correct"])
    print("Bin counts [schizo, healthy]:", probs["counts"])

def counter(resultA, resultB, actualresult):
    A_correct = 0
    A_wrong = 0
    B_correct = 0
    B_wrong = 0
    for i in range(len(actualresult)): # Checks for accuracy rate of each test
        if (resultA[i] == True and actualresult[i] == "schizophrenic") or (resultA[i] == False and actualresult[i] == 'healthy'):
            A_correct += 1
        else:
            A_wrong += 1
        
        if (resultB[i] == True and actualresult[i] == "schizophrenic") or (resultB[i] == False and actualresult[i] == 'healthy'):
            B_correct += 1
        else:
            B_wrong += 1
    return [A_correct, A_wrong, B_correct, B_wrong]

def joint_result_probs(resultA, resultB, actualresult):
    # Outcome order: AB, AB', A'B, A'B'
    # Each entry: [n_schizophrenic, n_healthy]
    bins = [[0, 0], [0, 0], [0, 0], [0, 0]]
    for i in range(len(actualresult)):
        A = resultA[i]
        B = resultB[i]
        if A and B:
            idx = 0
        elif A and not B:
            idx = 1
        elif not A and B:
            idx = 2
        else:
            idx = 3
        if actualresult[i] == "schizophrenic":
            bins[idx][0] += 1
        else:
            bins[idx][1] += 1
    
    p_schizo = []
    p_correct = []
    for schizo_n, healthy_n in bins:
        total = schizo_n + healthy_n
        if total == 0:
            p_schizo.append(0.0)
            p_correct.append(0.0)
        else:
            p_s = schizo_n / total
            p_schizo.append(p_s)
            # MAP diagnosis: pick majority class in this result bin
            p_correct.append(max(p_s, 1 - p_s))
    return {"p_schizo": p_schizo, "p_correct": p_correct, "counts": bins}

def testA(row):
    mg,decay,j = row
    if m.fabs(decay) > (2.47608430e-07)*mg**2 + (3.59130201e-03)*mg + (1.11493342e-01): # THIS IS BECAUSE DECAY IS NEGATIVE BUT IT NEEDS TO BE MADE POSITIVE
        return True
    else:
        return False

def testB(row):
    mg,decay,j = row
    if j < (8.07675322e-08)*mg**2 + (-1.43973630e-04)*mg + (7.00188346e-02):
        return True
    else:
        return False

def buildarraytest(var,statstr): # Add the variable you are looking for and the state of the person and get the values of it from everyone
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
    with open("test_param_values.csv",'r') as paramvalues: # like the buildarray, but for the test data
        values = csv.reader(paramvalues)
        for row in values:
            rawinput.append(row)
    matrix = rawinput[1:] # Truncated top row of matrix
    num = keydict[var]
    for row in matrix:
        if row[6] == statstr:
            output.append(row[num])
    return output

"""
The tests

Test A:
line:  m.fabs(decay) > (2.47608430e-07)*mg**2 + (3.59130201e-03)*mg + (1.11493342e-01) => Positive result

Test B:
line:  j < (8.07675322e-08)*mg**2 + (-1.43973630e-04)*mg + (7.00188346e-02) => Positive result

"""
if __name__ == "__main__":
    main()