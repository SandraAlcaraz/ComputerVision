import numpy as np
import cv2

def sobel (img):
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
    sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    return sobel_8u

def canny (img):
    filter = cv2.Canny(img,100,200)
    return filter

def prewit (img_gaussian):
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
    img_out= img_prewittx + img_prewitty
    return img_out

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    brightness = 10
    contrast = 200
    frame = np.int16(frame)
    frame = frame * (contrast/127+1) - contrast + brightness
    frame = np.clip(frame, 0, 255)
    change = np.uint8(frame)
    change = cv2.cvtColor(change, cv2.COLOR_BGR2GRAY)
    change = cv2.GaussianBlur(change,(9,9),cv2.BORDER_DEFAULT)

    (thresh, change) = cv2.threshold(change, 127, 255, cv2.THRESH_BINARY)
    change = prewit(change)
    #change = sobel(change)
    #change = canny(change)
    # Display the resulting frame
    cv2.imshow('frame',change)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
