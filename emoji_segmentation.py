import os
import numpy as np
from cv2 import cv2
from skin_detector.cv_helpers import cv2_show_img, plt_show_img, start_cv_video

RGB_RED = (255, 0, 0)
RGB_BLUE = (0, 255, 0)
RGB_GREEN = (0, 0, 255)

def filter(img, *params):
    original = img.copy()
    kernel = np.ones((3,3),np.uint8)

    img = apply_sobel(img)
    img = grayscale_binarization(img, threshold=10, bin_val=255)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, 2)

    circles = obtain_circle_positions(img)
    export_circle_regions(original, circles, 5)
    
    # result = draw_contours(img)
    result = draw_circles(img, original, 'squares', 5)

    return result

def draw_contours(img):
    # TODO: Maybe use average or std dev of areas to filter the contours

    output = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_cnt = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        perimeter_points = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
        area = cv2.contourArea(cnt)
        if (10 < len(perimeter_points) < 20) and (1000 < area < 90000):
            filtered_cnt.append(cnt)
            cv2.drawContours(output, perimeter_points, -1, RGB_BLUE, 10)

    cv2.drawContours(output, filtered_cnt, -1, RGB_RED, 2)
    return output

def export_circle_regions(img, circles, shift=0):
    try:
        os.mkdir('roi/')
    except Exception as e:
        print(e)
    for i, (x, y, r) in enumerate(circles):
        top = y-r-shift if y-r-shift >= 0 else 0
        bottom = y+r+shift if y+r+shift < img.shape[0] else img.shape[0]-1
        left = x-r-shift if x-r-shift >= 0 else 0
        right = x+r+shift if x+r+shift < img.shape[1] else img.shape[1]-1
        crop = img[top:bottom,left:right].copy()
        try:
            bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join('roi', f'{i}.jpg'), bgr)
        except Exception as e:
            print(e)
        

def obtain_circle_positions(img):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, minDist=100)#minDist=200, param1=30, param2=45, minRadius=0, maxRadius=0)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
    return circles
    
def draw_circles(img, original, type='circles', shift=0):
    circles = obtain_circle_positions(img)
    if circles is not None:
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            if type == 'squares':
                cv2.rectangle(original, (x - r - shift, y - r - shift), (x + r + shift, y + r + shift), RGB_GREEN, 2)
            else:
                cv2.circle(original, (x, y), r, RGB_GREEN, 4)
            cv2.rectangle(original, (x - 5, y - 5), (x + 5, y + 5), RGB_GREEN, -1)
    
    return original

def increase_sat(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype('float32')
    h, s, v = cv2.split(img)
    s = s + 200
    s = np.clip(s,0,255)
    imghsv = cv2.merge((h,s,v))
    imghsv = cv2.cvtColor(imghsv.astype('uint8'), cv2.COLOR_HSV2RGB)

    return imghsv

def grayscale_binarization(img, threshold=127, bin_val=255): 
    _ , img = cv2.threshold(img, threshold, bin_val, cv2.THRESH_BINARY)
    return img

def apply_sobel(img, *params):
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0
    kernel_size = 3

    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=kernel_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=kernel_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    new_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    #new_image = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)

    return new_image

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Emoji segmentation')
    parser.add_argument('-i', '--image', type=str, action='store', dest='src_img', help='The image to apply the filter')
    parser.add_argument('-ks', '--kernel_size', type=int, default=5, action='store', dest='kernel_size', help='The size of the kernel when applying dilatation or erotion')
    parser.add_argument('-d', '--dilate', type=int, default=0, action='store', dest='dilate_iter', help='Number of times to apply the Dilate operation')
    parser.add_argument('-e', '--erode', type=int, default=0, action='store', dest='erode_iter', help='Number of times to apply the Erode operation')
    args = parser.parse_args()

    if args.src_img:
        try:
            img = cv2.imread(args.src_img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_img = filter(img)
            plt_show_img(rgb_img)

        except Exception as error:
            print(error)

    else:
        start_cv_video(0, filter)
