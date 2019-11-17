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
    import argparse
    import glob

    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('-p', type=str, action='store', dest='path', required=True, help='The path to read all images')
    parser.add_argument('--tag', type=str, action='store', dest='tag', required=True, help='Tag name')
    args = parser.parse_args()
    
    df = None

    try:
        for file in glob.glob(f'{args.path}*.jpg'):
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if img.shape[0] != img.shape[1]: continue # Skip non-square images
            resized = resize_image(img)
            features = obtain_hog_features(resized)

            if df is None:
                df = pd.DataFrame(columns=[f'X{i}' for i in list(range(features.shape[0]))])

            df.loc[file] = features

        df.to_csv(f'{args.tag}_features.csv', header=True)

    except Exception as e:
        print(e)
