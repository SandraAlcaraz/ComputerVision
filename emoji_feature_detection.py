import os
import csv
import pandas as pd
from cv2 import cv2

def obtain_hog_features(img):
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    return h.flatten()

def resize_image(img, d=130):
    dim = (d, d)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

if __name__ == '__main__':
    
    df = None

    try:
        for file in os.listdir('roi/'):
            if file.endswith('.jpg'):
                img = cv2.imread(os.path.join('roi', file), cv2.IMREAD_COLOR)
                # Skip non-square images
                if img.shape[0] != img.shape[1]:
                    continue
                resized = resize_image(img)
                features = obtain_hog_features(resized)
                if df is None:
                    df = pd.DataFrame(columns=list(range(features.shape[0])))
                df.loc[file] = features
        df.to_csv('features.csv', header=False)

    except Exception as e:
        print(e)
