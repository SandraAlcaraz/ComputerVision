from emoji_segmentation import emoji_segmentation, get_circle_regions
from skin_detector.cv_helpers import start_cv_video, plt_show_img

from cv2 import cv2
import numpy as np
import keras

EMOJI_1 = None

EMOJI_DICT = {}

EMOJI_MODEL = keras.models.load_model('filter_model.h5')

def classify_emoji(img):
    res = EMOJI_MODEL.predict(np.array([img]) / 255 )
    print(res)
    plt_show_img(img)
    # calculate Hog
    # use classifer
    # ML Model

    return 1

def classify_mouth(img):
    images = extract_face_features(img)
    for img in images:
        img = resize_image(img, 100)
        # ML Model

    return 1

def extract_face_features(img):
    original = img.copy()
    #plt_show_img(img)
    img = process_emoji(img)
    
    filtered_cnt, max_cnt = get_contours(img)
    new_filter = get_inside_face_cnt(filtered_cnt, max_cnt)
    emoji_fts = []
    #mouth_emoji2 = get_mouth(new_filter)

    for mouth_emoji in new_filter:
        M = cv2.moments(mouth_emoji)
        cy = int(M['m01']/M['m00'])
        x,y,w,h = cv2.boundingRect(mouth_emoji)
        roi = original[y:y+h, x:x+w].copy()
        emoji_fts.append(roi)
        #plt_show_img(roi)
    
    return emoji_fts

def get_mouth(emoji_cnts):
    lower_y = -10
    lower_cnt = None
    for cnt in emoji_cnts:
        M = cv2.moments(cnt)
        cy = int(M['m01']/M['m00'])
        #print(cy)
        if cy > lower_y:
            lower_y = cy
            lower_cnt = cnt
    return lower_cnt

def get_inside_face_cnt(emoji_cnts, face_cnt):
    filtered = []
    max_x, max_y , max_w, max_h = cv2.boundingRect(face_cnt)
    for cnt in emoji_cnts:
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if cx > max_x and cx < max_w and cy > max_y and cy < max_h:
            filtered.append(cnt)
    
    return filtered

def get_contours(img):
    contours, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filtered_cnt = []
    max_area = -1
    max_cnt = None

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 100 and hier[0][i][3] == -1:
            if area > max_area:
                max_area = area
                max_cnt = cnt
            filtered_cnt.append(cnt)

    filtered_cnt.remove(max_cnt)
    
    return filtered_cnt, max_cnt

def draw_emoji(frame, emoji_index, emoji_pos):
    print('Drawing...')
    real_emoji, inverse_mask = EMOJI_DICT.get(emoji_index)
    x, y, r = emoji_pos
    top = y - r
    bottom = y + r
    left = x - r
    rigth = x + r

    emoji = cv2.resize(real_emoji, (rigth - left, bottom - top))
    inverse_mask = cv2.resize(inverse_mask, (rigth - left, bottom - top))
    overlap_area = frame[top:bottom, left:rigth]
    overlap_area = cv2.bitwise_and(overlap_area, overlap_area, mask=inverse_mask)
    overlap_area = cv2.add(overlap_area, emoji)
    frame[top:bottom, left:rigth] = overlap_area

    return frame

def detect_emoji(frame):
    circles, mask, _ = emoji_segmentation(frame)
    possible_emojis = get_circle_regions(frame, circles, 1.2)
    
    for emoji in possible_emojis:
        cropped_img, cropped_pos = emoji
        if not is_square(cropped_img): continue
        cropped_img = resize_image(cropped_img)
        is_emoji = classify_emoji(cropped_img)
        if not is_emoji: continue
        emoji_type = classify_mouth(cropped_img)
        draw_emoji(frame, emoji_type, cropped_pos)

    return frame# * mask

def is_square(img):
    return img.shape[0] == img.shape[1]

def process_emoji(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3,3),np.uint8)

    img = apply_sobel(img)
    img = grayscale_binarization(img, threshold=10, bin_val=255)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, 1)
    
    return img

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

def resize_image(img, d=350):
    dim = (d, d)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Emoji segmentation')
    parser.add_argument('-i', '--image', type=str, action='store', dest='src_img', required=True, help='The input image')
    args = parser.parse_args()

    try:
        for i in range(1, 7):
            original = cv2.imread(f'emojis/{i}.png', cv2.IMREAD_UNCHANGED)
            r, g, b, a = cv2.split(original)
            emoji = cv2.merge((r,g,b)).astype('uint8')
            mask = a.astype('uint8')
            inverse_mask = cv2.bitwise_not(mask)
            emoji = cv2.bitwise_and(emoji, emoji, mask=mask)
            EMOJI_DICT[i] = emoji, inverse_mask

        img = cv2.imread(args.src_img, cv2.IMREAD_COLOR)
        plt_show_img(detect_emoji(img))
        #start_cv_video(0, img_filter=detect_emoji)

    except Exception as error:
        print(error)