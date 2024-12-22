
import numpy as np
#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import pandas as pd
import sklearn
######################################################################
# kNN classifier for rain/total bikes on bridges
##################################################################

if __name__ == "__main__":
   
    unfiltered_bike = pd.read_csv("NYC_Bicycle_Counts_2016_Corrected.csv")
    unfiltered_bike = unfiltered_bike.to_numpy().tolist()
    
    filtered_bike = []
    for i in range(len(unfiltered_bike)):
        aDay = unfiltered_bike[i]
        if not (('S' in aDay[4]) or ('T' in aDay[4])):
            aDay[4] = float(aDay[4])
            if aDay[4] > 0: # settin not raining to 0 and raining to 1 in data
                aDay[4] = 1
            for j in range(5,10):
                aDay[j] = int(aDay[j].replace(',','' ))
            filtered_bike.append(aDay)
        
    bikearray = np.array(filtered_bike)
    
    X = bikearray[:,9].astype('float64').reshape(-1,1)
    y = bikearray[:,4].astype('float64')
    accuracy = [0,0,0,0,0]
    for j in range(1,101):
        [train_data, test_data, train_labels, test_labels] = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=j)
        ### fill in any necessary code below to perform the task outlined in README.md document. ###
        for i in range(1,6):
        # create KNN classifier
            knn = KNeighborsClassifier(n_neighbors=i)
        
            # train the model using the training sets
            knn.fit(train_data,train_labels)
        
            # predict the response for test dataset
            test_pred = knn.predict(test_data)
            accuracy[i-1] += metrics.accuracy_score(test_labels,test_pred)
        # Model Accuracy, how often is the classifier correct?
    for i in range(1,6):
        print("k =",i)
        print("Accuracy:",accuracy[i-1]/100)
    
        # calcuating the confusion matrix
        confusion_matrix = metrics.confusion_matrix(test_labels,test_pred)
        print(confusion_matrix)
        print('')
