from emoji_segmentation import emoji_segmentation, get_circle_regions
from skin_detector.cv_helpers import start_cv_video, plt_show_img

from cv2 import cv2
import numpy as np

EMOJI_1 = None

EMOJI_DICT = {}

def classify_emoji(img):
    # calculate Hog
    # use classifer
    return 1

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
        # Crop mouth if it is used HERE!
        emoji_type = classify_emoji(cropped_img)
        if emoji_type > 0:
            draw_emoji(frame, emoji_type, cropped_pos)

    return frame# * mask

def is_square(img):
    return img.shape[0] == img.shape[1]

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
        #plt_show_img(detect_emoji(img))
        start_cv_video(0, img_filter=detect_emoji)

    except Exception as error:
        print(error)