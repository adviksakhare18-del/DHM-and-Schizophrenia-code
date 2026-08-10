# 3. THIS IS THE FUNCTION THAT ADDS THE DECAY AND FREQUENCY CONSTANTS TO THE CSV VALUES
import mpmath as m
from useful import *
import csv

def main():
    rawmatrix = []
    with open("param_values_both.csv",'r') as read1:
        read = csv.reader(read1)
        for row in read:
            rawmatrix.append(row)
    matrix = rawmatrix[1:]
    newmatrix = addons(matrix)
    
    with open('param_values_final.csv','a',newline="") as final1:
        final = csv.writer(final1)
        for series in newmatrix:
            final.writerow(series)
    print("Success!")

rowdict = {
    "mg":0,
    "b":1,
    "j":2,
    "k":3,
    "l":4,
    "base":5,
    "health":6
}

def addons(matrix):
    output = []
    output.append(['mg','b','j','k','l','base','health','decay','frequency'])
    for line in matrix:
        for i in range(0,len(line)):
            try:
                line[i] = float(line[i])
            except (ValueError, TypeError):
                pass
        freq = m.sqrt((4*line[2]*line[4] - line[3]**2)/(4*line[2]**2))
        decay = line[3]/(2*line[2])
        output.append(arraymerger([line,[decay,freq]]))
    return output

# Q =  sqrt((4jl−k²)/4j²) (oscillation frequency)
# decay = k/(2j) (decay rate)

if __name__ == "__main__":
    main()