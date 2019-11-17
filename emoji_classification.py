from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import pandas as pd
import os

if __name__ == "__main__":

    #pd.read_csv('features/0_features.csv')

    training_dir = os.listdir('features/')
    features_df = [None] * 7
    
    for file in training_dir:
        i, _ = file.split('_')
        features_df[int(i)] = pd.read_csv('features/'+ file, index_col = 0)

    X = pd.concat(features_df, ignore_index=True)
    print(X)