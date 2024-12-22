# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:48:53 2021

@author: Ben
"""
#resusing hw to determine regression of given data
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import matplotlib.pyplot as plt
import sklearn

def main():
    
    unfiltered_bike = pd.read_csv("NYC_Bicycle_Counts_2016_Corrected.csv")
    unfiltered_bike = unfiltered_bike.to_numpy().tolist()
    
    filtered_bike = []
    for i in range(len(unfiltered_bike)):
        aDay = unfiltered_bike[i]
        if not (('S' in aDay[4]) or ('T' in aDay[4])):
            aDay[4] = float(aDay[4])
            for j in range(5,10):
                aDay[j] = int(aDay[j].replace(',','' ))
            filtered_bike.append(aDay)
        
    bikearray = np.array(filtered_bike)
    
    X = bikearray[:,2:5].astype('float64')
    y = bikearray[:,9].astype('float64')
    rsq = 0
    x1 =0
    x2=0
    x3=0
    myyint =0
    for j in range(1,101):
        [X_train, X_test, y_train, y_test] = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=j)
        
        #Normalizing training and testing data
        [X_train, trn_mean, trn_std] = normalize_train(X_train)
        X_test = normalize_test(X_test, trn_mean, trn_std)
    
        #Define the range of lambda to test
        lmbda = [0] # np.logspace(-4,2,num=71)
    
        MODEL = []
        MSE = []
        for l in lmbda:
            #Train the regression model using a regularization parameter of l
            model = train_model(X_train,y_train,l)
    
            #Evaluate the MSE on the test set
            rsq += error(X_test,y_test,model)
            x1 += model.coef_[0]
            x2 += model.coef_[1]
            x3 += model.coef_[2]
            myyint += model.intercept_
            print(model.coef_)
            #Store the model and mse in lists for further processing
            MODEL.append(model)
            #MSE.append(mse)
    
    return [rsq/100, x1/100, x2/100, x3/100, myyint/100]


#Function that normalizes features in training set to zero mean and unit variance.
#Input: training data X_train
#Output: the normalized version of the feature matrix: X, the mean of each column in
#training set: trn_mean, the std dev of each column in training set: trn_std.
def normalize_train(X_train):

    #fill in
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    transpose = X_train.T
    X = X_train.T
    for i in range(len(transpose)):
        X[i] = (X[i] - mean[i]) / std[i] 
    X = X.T
    return X, mean, std


#Function that normalizes testing set according to mean and std of training set
#Input: testing data: X_test, mean of each column in training set: trn_mean, standard deviation of each
#column in training set: trn_std
#Output: X, the normalized version of the feature matrix, X_test.
def normalize_test(X_test, trn_mean, trn_std):

    #fill in
    transpose = X_test.T
    X = X_test.T
    for i in range(len(transpose)):
        X[i] = (X[i] - trn_mean[i]) / trn_std[i] 
    X = X.T

    return X



#Function that trains a ridge regression model on the input dataset with lambda=l.
#Input: Feature matrix X, target variable vector y, regularization parameter l.
#Output: model, a numpy object containing the trained model.
def train_model(X,y,l):
    ridgemod = LinearRegression()
    ridgemod.fit(X,y)


    return ridgemod

#Function that trains a Lasso regression model on the input dataset with lambda=l.
#Input: Feature matrix X, target variable vector y, regularization parameter l.
#Output: model, a numpy object containing the trained model
def train_model_lasso(X,y,l):

    lassomod = Lasso(alpha=l, fit_intercept=True)
    lassomod.fit(X,y)

    return lassomod

#Function that calculates the r squared error the model on the input dataset.
#Input: Feature matrix X, target variable vector y, numpy model object
#Output: r squared
def error(X,y,model):
    ymodel = model.predict(X)
    #mse = sklearn.metrics.mean_squared_error(y,ymodel)
    return (sklearn.metrics.r2_score(y,ymodel))
    #Fill in

if __name__ == '__main__':
    
    model_best = main()
    print(model_best)
    #We use the following functions to obtain the model parameters instead of model_best.get_params()
    #print(model_best.coef_)
    #print(model_best.intercept_)
