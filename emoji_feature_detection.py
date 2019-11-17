import csv
import pandas as pd
import imutils
from cv2 import cv2

from skin_detector.cv_helpers import plt_show_img

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
    
    df = pd.DataFrame(columns=list(range(680400)))

    try:
        for file in glob.glob(f'{args.path}*.jpg'):
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if img.shape[0] != img.shape[1]: continue # Skip non-square images
            resized = resize_image(img, 200)
            print('after resize')
            #left = imutils.rotate(resized, 30)
            #plt_show_img(resized)
            features = obtain_hog_features(resized)
            df = df.append(pd.Series(features), ignore_index=True)

        df.to_csv(f'{args.tag}_features.csv', header=False)

    except Exception as e:
        print(e)
