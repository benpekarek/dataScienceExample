# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:53:40 2021

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

#Return fitted model parameters to the dataset at datapath for each choice in degrees.
#Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
#Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
#coefficients when fitting a polynomial of d = degrees[i].
def main(datapath, degrees):
    unfiltered_bike = pd.read_csv("NYC_Bicycle_Counts_2016_Corrected.csv")
    unfiltered_bike = unfiltered_bike.to_numpy().tolist()

    filtered_bike = []
    for i in range(len(unfiltered_bike)):
        aDay = unfiltered_bike[i]
        if not (('S' in aDay[4]) or ('T' in aDay[4])):
            aDay[4] = float(aDay[4])
            for j in range(5,10):
                aDay[j] = int(aDay[j].replace(',','' ))
            for j in range(2,5):
                aDay[j] = float(aDay[j])
            filtered_bike.append(aDay)
    bridges = ['Brooklyn', 'Manhattan', 'Williamsburg', 'Queensboro']
    for i in range(5,9):
        xcol = []
        ycol = []
        for row in filtered_bike:
            xcol.append(float(row[i]))
            ycol.append(float(row[-1]))
        paramFits = [least_squares(feature_matrix(xcol,d),ycol) for d in degrees]
        #fill in
        #read the input file, assuming it has two columns, where each row is of the form [x y] as
        #in poly.txt.
        #iterate through each n in degrees, calling the feature_matrix and least_squares functions to solve
        #for the model parameters in each case. Append the result to paramFits each time.
        plt.figure()
        plt.scatter(xcol, ycol)
        t = np.arange(0, 10000, 1)
        plt.plot(t, paramFits[0][1] + paramFits[0][0] *t, 'c--')
        
        plt.legend(['1st degree','data'])
        rsquared = sklearn.metrics.r2_score(ycol, [paramFits[0][1] + paramFits[0][0] *t for t in xcol] )
        plt.title(bridges[i-5] + ' Bridge, r^2 = ' + f"{rsquared:.2f}")
        plt.xlabel('Number of Bikers on this Bridge')
        plt.ylabel('Total Bikers on All Bridges')
        plt.show()
        print(paramFits)
        print(sklearn.metrics.r2_score(ycol,[paramFits[0][1] + paramFits[0][0] *t for t in xcol]))
        #print(sklearn.metrics.r2_score(ycol,[paramFits[1][2] + paramFits[1][1] *t + paramFits[1][0]* t**2 for t in xcol]))
    return paramFits


#Return the feature matrix for fitting a polynomial of degree d based on the explanatory variable
#samples in x.
#Input: x as a list of the independent variable samples, and d as an integer.
#Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
#for the ith sample. Viewed as a matrix, X should have dimension #samples by d+1.
def feature_matrix(x, d):

    #fill in
    #There are several ways to write this function. The most efficient would be a nested list comprehension
    #which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    return [[a ** d for d in range(d,-1,-1)] for a in x]


#Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
#Input: X as a list of features for each sample, and y as a list of target variable samples.
#Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)

    #fill in
    #Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.

    return np.linalg.inv(X.T @ X) @ X.T @ y

if __name__ == '__main__':
    datapath = 'poly.txt'
    degrees = [1,2,3, 4,5]

    paramFits = main(datapath, degrees)
    if (len(paramFits) == 5):
        pass#print(paramFits[2][3] + paramFits[2][2] *2 + paramFits[2][1]* 2**2 + paramFits[2][0] * 2** 3)
