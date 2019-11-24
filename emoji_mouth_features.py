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

from skin_detector.cv_helpers import plt_show_img

def grayscale_binarization(img, threshold=127, bin_val=255): 
    _ , img = cv2.threshold(img, threshold, bin_val, cv2.THRESH_BINARY)
    return img

def apply_sobel(img, *params):
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0
    kernel_size = 3

    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=kernel_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=kernel_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    new_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    #new_image = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)

    return new_image

def process_image(img):
    kernel = np.ones((3,3),np.uint8)

    img = apply_sobel(img)
    img = grayscale_binarization(img, threshold=10, bin_val=255)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, 1)
    
    return img

def obtain_hog_features(img):
    hog = cv2.HOGDescriptor()
    winStride = (4,4)
    padding = (4,4)
    locations = ((5,10),)
    #h = hog.compute(img,winStride,padding,locations)
    h = hog.compute(img)
    return h.flatten()

def resize_image(img, d=130, d2=130):
    dim = (d, d2)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def get_dataframes():
    X = []
    X_SVM = []
    Y = []
    Y_SVM = []

    try:
        for i in range(1, 7):
            print('reading: ' + str(i))
            for file in glob.glob(f'roi_emoji/{i}/*.jpg'):
                print(file)
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                #img = process_image(img)
                img = resize_image(img, 700,500)
                #plt_show_img(img)
                features = obtain_hog_features(img)

                if i > 0:
                    X.append(features)
                    Y.append(i)
                X_SVM.append(features)
                Y_SVM.append(0 if i == 0 else 1)

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
    
    test_size = 0.2
    seed = 5
    #X_SVM_train, X_SVM_test, Y_SVM_train, Y_SVM_test = train_test_split(X_SVM, Y_SVM, test_size=test_size, random_state=seed)
    
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
    
    multiclass_filename = 'emoji_tree_model_multiclass.sav'
    if not args.retrain and 'emoji_tree_model_multiclass.sav' in os.listdir():
        dtree = pickle.load(open(multiclass_filename, 'rb'))
    else:
        dtree_model = DecisionTreeClassifier()
        dtree = dtree_model.fit(X_train, Y_train)
        pickle.dump(dtree_model, open(multiclass_filename, 'wb'))
    
    dtree_pred = dtree.predict(X_test)
    print(Y_test)
    print(dtree_pred)
    

    
    
    
