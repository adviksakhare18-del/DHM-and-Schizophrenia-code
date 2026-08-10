# 6. THIS IS THE FILE THAT CALCULATES THE BAYES' ADJUSTED PROBABILITY OF HAVING SCHIZOPHRENIA FOR 4 COMBINATIONS OF BOTH TESTS.
# This is on the real dataset, not on the test one
from useful import *
from Series_plot import buildarray


"""
Criteria & Equations:
These tests will be called A and B respectively, with A being positive and A' being 
negative for schizophrenia according to the test.

- mg vs decay: Parabola: (2.47608430e-07)x^2 + (3.59130201e-03)x + (1.11493342e-01)
the most promising 
A = above
A' = below

- mg vs j: Parabola: (8.07675322e-08)x^2 + (-1.43973630e-04)x + (7.00188346e-02)
second most promising
B = below
B' = above

Outcome order: [ab, ab', a'b, a'b']  (A/B test results, not schizophrenia status)
"""

PSCHIZO = 0.01  # P(schizophrenic) for a randomly chosen Australian

def main():
    # mg setup
    mghealthy = multifunc(buildarray("mg","healthy"),float) # Array of mg given healthy
    mgschizo = multifunc(buildarray('mg','schizophrenic'),float) # Array of mg given schizo
    mg = arraymerger([mghealthy,mgschizo]) # Array of mg for all patients
    # j setup
    jhealthy = multifunc(buildarray("j","healthy"),float) # Array of j given healthy
    jschizo = multifunc(buildarray('j','schizophrenic'),float) # Array of j given schizo
    j = arraymerger([jhealthy,jschizo]) # Array of j for all patients
    # decay setup
    decayhealthy = multifunc(buildarray("decay","healthy"),float) # Array of decay given healthy
    decayschizo = multifunc(buildarray('decay','schizophrenic'),float) # Array of decay given schizo
    decay = arraymerger([decayhealthy,decayschizo]) # Array of decay for all patients
    # Making information matrices (each row: mg, j, decay)
    initmatrixhealthy = mattranspose([mghealthy, jhealthy, decayhealthy])
    initmatrixschizo = mattranspose([mgschizo, jschizo, decayschizo])
    initmatrixall = mattranspose([mg, j, decay])

    condgivenhealthy = probfinder(initmatrixhealthy)
    condgivenschizo = probfinder(initmatrixschizo)
    condall = probfinder(initmatrixall)  # P(cond) in the 50/50 study sample only

    labels = "[ab, ab', a'b, a'b']"
    print(f"P(cond | healthy) {labels}: {condgivenhealthy}")
    print(f"P(cond | schizo)  {labels}: {condgivenschizo}")
    print(f"P(cond) in study sample {labels}: {condall}\n")

    schizogivencond = bayes(condgivenschizo, condgivenhealthy)
    print(f"P(schizo | cond) {labels} (population, P(S)={PSCHIZO}): {schizogivencond}")

# We can treat this as 2 tests, with 4 cumulative outcomes, and find P(+|cond)
# Each cond. is an outcome of both tests

def bayes(condgivenschizo, condgivenhealthy, pschizo=PSCHIZO):
    """P(S | cond) = P(cond | S) * P(S) / P(cond), with population P(cond)."""
    output = []
    phealthy = 1 - pschizo
    for i in range(len(condgivenschizo)):
        p_cond = condgivenschizo[i] * pschizo + condgivenhealthy[i] * phealthy
        if p_cond == 0:
            output.append(0.0)
        else:
            output.append(condgivenschizo[i] * pschizo / p_cond)
    return output

def probfinder(initmatrix): # row order: mg j decay
    length = len(mattranspose(initmatrix)[0])
    ab = 0
    anotb = 0
    notab = 0
    notanotb = 0
    for row in initmatrix:
        mg,j,decay = row
        A = checkA(mg,decay)
        B = checkB(mg,j)
        if A and B:
            ab += 1
        elif A and not B:
            anotb += 1
        elif not A and B:
            notab += 1
        elif not A and not B:
            notanotb += 1
    return [ab/length, anotb/length, notab/length, notanotb/length]

# checkA and checkB are the functions that run tests A and B on the given datapoints
def checkA(mg,decay):
    if decay > (2.47608430e-07)*mg**2 + (3.59130201e-03)*mg + (1.11493342e-01):
        return True
    return False

def checkB(mg,j):
    if j < (8.07675322e-08)*mg**2 + (-1.43973630e-04)*mg + (7.00188346e-02):
        return True
    return False

if __name__ == "__main__":
    main()