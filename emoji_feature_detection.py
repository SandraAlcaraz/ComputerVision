import csv
import imutils
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cv2 import cv2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.utils.multiclass import unique_labels

def obtain_hog_features(img):
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    return h.flatten()

def resize_image(img, d=130):
    dim = (d, d)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def get_dataframes():
    # X = pd.DataFrame(columns=list(range(680400)), dtype='float64')
    X = []
    X_SVM = []
    Y = []
    Y_SVM = []

    try:
        for i in range(0, 7):
            print('reading: ' + str(i))
            for file in glob.glob(f'roi/{i}/*.jpg'):
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                if img.shape[0] != img.shape[1]: continue # Skip non-square images
                resized = resize_image(img, 200)
                #left = imutils.rotate(resized, 30)
                #plt_show_img(resized)
                features = obtain_hog_features(resized)
                # X = X.append(pd.Series(features), ignore_index=True)
                if i > 0:
                    X.append(features)
                    Y.append(i)
                X_SVM.append(features)
                Y_SVM.append(0 if i == 0 else 1)

            # X.to_csv(f'{args.tag}_features.csv', header=False)
    except Exception as e:
        print(e)
    return X, X_SVM, Y, Y_SVM

if __name__ == '__main__':
    import argparse
    import glob

    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('-t', action='store_true', dest='retrain', help='Retrain?')
    # parser.add_argument('-p', type=str, action='store', dest='path', required=True, help='The path to read all images')
    # parser.add_argument('--tag', type=str, action='store', dest='tag', required=True, help='Tag name')
    args = parser.parse_args()
    
    X, X_SVM, Y, Y_SVM = get_dataframes()
    
    test_size = 0.33
    seed = 5
    X_SVM_train, X_SVM_test, Y_SVM_train, Y_SVM_test = train_test_split(X_SVM, Y_SVM, test_size=test_size, random_state=seed)
    
    # svm_filename = 'svm_model.sav'
    # if not args.retrain and 'svm_model.sav' in os.listdir():
    #     svm = pickle.load(open(svm_filename, 'rb'))
    # else:
    #     svm_model = SVC(kernel='rbf')
    #     svm = svm_model.fit(X_SVM_train, Y_SVM_train)
    #     pickle.dump(svm_model, open(svm_filename, 'wb'))
    
    # svm_pred = svm.predict(X_SVM_test)
    # print(Y_SVM_test)
    # print(svm_pred)
    
    # tree_filename = 'tree_model.sav'
    # if not args.retrain and 'tree_model.sav' in os.listdir():
    #     dtree = pickle.load(open(tree_filename, 'rb'))
    # else:
    #     dtree_model = DecisionTreeClassifier(class_weight='balanced')
    #     dtree = dtree_model.fit(X_SVM_train, Y_SVM_train)
    #     pickle.dump(dtree_model, open(tree_filename, 'wb'))
    
    # dtree_pred = dtree.predict(X_SVM_test)
    # print(Y_SVM_test)
    # print(dtree_pred)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    multiclass_filename = 'tree_model_multiclass.sav'
    if not args.retrain and 'tree_model_multiclass.sav' in os.listdir():
        dtree = pickle.load(open(multiclass_filename, 'rb'))
    else:
        dtree_model = DecisionTreeClassifier()
        dtree = dtree_model.fit(X_train, Y_train)
        pickle.dump(dtree_model, open(multiclass_filename, 'wb'))
    
    dtree_pred = dtree.predict(X_test)
    print(Y_test)
    print(dtree_pred)
    
    

    
    
    
