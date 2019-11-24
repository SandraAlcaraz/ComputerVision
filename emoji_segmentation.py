import os
import numpy as np
from cv2 import cv2
from skin_detector.cv_helpers import plt_show_img, start_cv_video

RGB_RED = (255, 0, 0)
RGB_BLUE = (0, 255, 0)
RGB_GREEN = (0, 0, 255)

def emoji_segmentation(img):
    """Finds circles in an image.

    Parameters:
        img -- The input image

    Returns:
        List[(int, int, int)] -- List of center and radius of found circles (x,y,r)
        binary image -- mask with areas of found circles 
        image -- Original image with found circles drawn on it
    """
    original = img.copy()
    kernel = np.ones((3,3),np.uint8)

    img = apply_sobel(img)
    img = grayscale_binarization(img, threshold=10, bin_val=255)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, 2)

    circles = obtain_circle_positions(img)
    drawn_circles, mask = draw_circles(circles, original, 'squares', 1.2)
    
    return circles, mask, drawn_circles

def get_circle_regions(img, circles, shift=1):
    """Crops areas according to the positions of the given circles.

    Parameters:
        img -- The input image
        circles -- List[(int, int, int)]: The list of positions and radius of circles
        shift -- int: the radius multiplier (default=1)
    
    Return:
        List[(image, (int,int,int))] -- A list of cropped images with its coordinates (x,y,r)
    """
    crops = []
    if not circles: return []
    for circle in circles:
        x, y, r = circle
        r = int(r * shift)
        top = y-r if y-r >= 0 else 0
        bottom = y+r if y+r < img.shape[0] else img.shape[0]-1
        left = x-r if x-r >= 0 else 0
        right = x+r if x+r < img.shape[1] else img.shape[1]-1
        crop = img[top:bottom,left:right].copy()
        crops.append((crop, circle))

    return crops

def export_circle_regions(regions, j):
    try:
        os.mkdir('roi/')
    except Exception as e:
        print(e)
    
    n = len(os.listdir('roi/'))
    for i, crop in enumerate(regions):
        try:
            img, _ = crop
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join('roi', f'{i + n}_{j}.jpg'), bgr)
        except Exception as e:
            print(e)
        

def obtain_circle_positions(img):
    try:
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.8, minDist=90, maxRadius=250, minRadius=20)#minDist=200, param1=30, param2=45, minRadius=0, maxRadius=0)
    except Exception as e:
        print(e)

    print("Finding circles...")
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
    return circles
    
def draw_circles(circles, original, type='circles', shift=1):
    mask = np.zeros((original.shape[0], original.shape[1], 3), 'uint8')
    if circles is not None:
        for (x, y, r) in circles:
            r = int(r * shift)
            cv2.circle(mask, (x, y), r, (1,1,1), -1)
            if type == 'squares':
                cv2.rectangle(original, (x - r, y - r), (x + r, y + r), RGB_BLUE, 3)
            else:
                cv2.circle(original, (x, y), r, RGB_GREEN, 4)
            cv2.rectangle(original, (x - 5, y - 5), (x + 5, y + 5), RGB_GREEN, -1)
    
    return original, mask

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
    parser.add_argument('-i', '--image', type=str, action='store', dest='src_img', required=True, help='The input image')
    parser.add_argument('-s', '--save', action='store_true', dest='store', help='Store output?')
    args = parser.parse_args()

    try:
        img = cv2.imread(args.src_img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        circles, mask, drawn = emoji_segmentation(img)
        cropped_imgs = get_circle_regions(img, circles, 1.2)
        _, n = args.src_img.split('/')
        n, _ = n.split('.')
        if args.store:
            export_circle_regions(cropped_imgs, n)

        plt_show_img(drawn)

    except Exception as error:
        print(error)
