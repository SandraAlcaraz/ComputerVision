import numpy as np
import cv2
from matplotlib import pyplot as plt

def getMask(hsvImage):

    lowerRed1 = (0,55,40)
    upperRed1 = (20,255,255)
    maskRed1 = cv2.inRange(hsvImage, lowerRed1, upperRed1)
    lowerRed2 = (160,55,40)
    upperRed2 = (180,255,255)
    maskRed2 = cv2.inRange(hsvImage, lowerRed2, upperRed2)
    maskRed = maskRed1 + maskRed2
    
    return maskRed

def applyMask(mask, hsvImage):
    return cv2.bitwise_and(hsvImage, hsvImage, mask=mask)

def processFrame(cap):
    _, frame = cap.read()
    
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = getMask(hsvImage)

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

if __name__ == "__main__":
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