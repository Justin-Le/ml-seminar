"""

########################################
Preprocess a dataset stored in CSV file.
########################################

Extract numbers and discard text.
Convert values to floating-point.
Impute empty fields with mean values.

Input: filename (include .csv extension)
Return: numpy array of processed data

########################################
"""

import csv
import numpy as np
from sklearn.preprocessing import Imputer

def process_wdbc(filename):
    # Read dataset
    data = []
    with open(str(filename), 'rb') as csvfile:
        d = csv.reader(csvfile)
        for row in d:
            data.append(row)

    # Store variable names
    variables = data[0]

    # Remove variable names
    data = data[1:]

    # Print metadata
    print "Number of Examples: %i" % len(data)
    print "Number of Variables: %i \n" % len(data[0])
    print "Variables: \n %s \n" % ', '.join(variables)
    print "Raw example: \n %s \n" % ', '.join(data[1])

    for i in range(len(data)):
        for j in range(len(data[0])):
            s = data[i][j]

            # Set class labels (malignant or benign)
            if s == 'M':
                s_new = 1
            elif s == 'B':
                s_new = 0
            else:
                # Extract floating-point numbers
                s_new = ''.join(char for char in s if char.isdigit() or char == '.')

                # Insert NaN for sklearn's Imputer
                if s_new == '': 
                    s_new = np.nan
                
            data[i][j] = s_new

    # Impute missing values
    imp = Imputer()
    imp.fit(data)
    data = imp.transform(data)

    print "==== Data has been processed. ==== \n"
    print "New number of variables: %i \n" % len(data[0])
    print "Clean example: \n %s \n" % str(data[0])

    return np.array(data)

    """
    # Print min, max, and mean of each variable
    # Warning: only execute for small number of variables

    print "\n"
    for i in range(len(data[0])):
        print "Variable: %s" % variables[i]
        print "Min: %s" % min(data[:, i])
        print "Max: %s" % max(data[:, i])
        print "Mean: %s" % np.mean(data[:, i])
        print "\n"
"""

if __name__ == "__main__":
    print __doc__
