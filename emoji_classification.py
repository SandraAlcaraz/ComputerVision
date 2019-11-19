import cv2 as cv2
import numpy as np

from skin_detector.cv_helpers import plt_show_img

RGB_RED = (255, 0, 0)
RGB_BLUE = (0, 0, 255)
RGB_GREEN = (0, 255, 0)

def resize_image(img, d=130):
    dim = (d, d)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

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

def draw_contours(img):
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = process_image(img)

    contours, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filtered_cnt = []
    #print(hier)
    for i, cnt in enumerate(contours):
        #perimeter = cv2.arcLength(cnt, True)
        #perimeter_points = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
        area = cv2.contourArea(cnt)
        #M = cv2.moments(cnt)
        #cx = int(M['m10']/M['m00'])
        #cy = int(M['m01']/M['m00'])
        #print(hier[0][i])
        if area > 100 and hier[0][i][3] == -1: #(10 < len(perimeter_points) < 20):  
            print(area) 
            filtered_cnt.append(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x,y),(x+w,y+h), RGB_GREEN, 1)
            #ROI = image[y:y+h, x:x+w]
        #cv2.drawContours(output, perimeter_points, -1, RGB_BLUE, 10)

    cv2.drawContours(output, filtered_cnt, -1, RGB_RED, 1)
    return output

if __name__ == "__main__":
    img = cv2.imread('roi/6/35.jpg', cv2.IMREAD_GRAYSCALE)
    if img.shape[0] == img.shape[1]:
        img = resize_image(img, 200)
        img = draw_contours(img)
        plt_show_img(img)
