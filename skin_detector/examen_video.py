import numpy as np
import cv2
from matplotlib import pyplot as plt
from cv_helpers import plt_show_img, show_compared_imgs

def start_camera():
    cap = cv2.VideoCapture(0)
    plt.ion()

    while(cap.isOpened()):
        processFrame(cap)
        
        press = plt.waitforbuttonpress(0.01)
        if press is None or press == False:
            pass
        else:
            break

    cap.release()
    plt.ioff()
    plt.show()

def processFrame(cap):
    _, frame = cap.read()
    
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = get_skin_mask(hsvImage)

    kernel = np.ones((5,5),np.uint8)
    maskClosed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    segmentedImageHSV = applyMask(maskClosed, hsvImage)
    segmentedImage = cv2.cvtColor(segmentedImageHSV, cv2.COLOR_HSV2RGB)

    plt.title('Segmented')
    plt.subplot(2,1,1)
    plt.imshow(segmentedImage)
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,1,2)
    plt.title('Original')
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.xticks([]),plt.yticks([])

def get_skin_mask(hsvImage):
    lowerRed1 = (0,48,90)#50
    upperRed1 = (13,205,255)
    maskRed1 = cv2.inRange(hsvImage, lowerRed1, upperRed1)
    lowerRed2 = (170,45,80)
    upperRed2 = (180,255,255)
    #maskRed2 = cv2.inRange(hsvImage, lowerRed2, upperRed2)
    maskRed = maskRed1 #+ maskRed2
    
    return maskRed

def applyMask(mask, hsvImage):
    return cv2.bitwise_and(hsvImage, hsvImage, mask=mask)

def skin_binarization(hsv_image):
    '''
        Assumes HSV image
    '''
    k_size = 7
    gauss_size = 3
    iterations = 3

    hsv_image = cv2.GaussianBlur(hsv_image, (gauss_size, gauss_size), cv2.BORDER_DEFAULT)

    mask = get_skin_mask(hsv_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    #kernel = np.ones((k_size, k_size), np.uint8)
    maskOpened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    maskClosed = cv2.morphologyEx(maskOpened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    segmented_hsv = applyMask(maskClosed, hsv_image)
    new_img = cv2.cvtColor(segmented_hsv, cv2.COLOR_HSV2RGB)

    return new_img

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Skin Detector')
    parser.add_argument('-i', '--image', type=str, action='store', dest='src_img', help='The image to apply the filter')
    args = parser.parse_args()

    if args.src_img:
        try:
            img = cv2.imread(args.src_img, cv2.IMREAD_COLOR)
            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            new_img = skin_binarization(hsv_image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            show_compared_imgs(img, new_img)

        except Exception as error:
            print(error)

    else:
        start_camera()
