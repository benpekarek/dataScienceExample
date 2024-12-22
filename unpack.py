# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:43:55 2021

@author: Ben
"""

import pandas as pd
import numpy as np

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
    
bikearray = np.array(filtered_bike)

