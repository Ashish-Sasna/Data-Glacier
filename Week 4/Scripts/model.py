# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:42:18 2023

@author: Ashish S
"""

# Importing Libraries
import os
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Loading diabetes dataset
diab_data = load_diabetes(as_frame=True)
diab_df = pd.DataFrame(data=diab_data.data, 
                       columns=diab_data.feature_names)

diab_tar = pd.DataFrame(data=diab_data.target) 

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(diab_df, 
                                                    diab_tar, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Model training and prediction
lr = LinearRegression()
lr_model = lr.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

# Save the model
os.chdir('..')
pkl.dump(lr_model, open(os.path.join(os.getcwd(), 'model.pkl'), 'wb'))
