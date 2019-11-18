from emoji_segmentation import emoji_segmentation, get_circle_regions
from skin_detector.cv_helpers import start_cv_video, plt_show_img

from cv2 import cv2

EMOJI_1 = None

def draw_emoji(frame, real_emoji, emoji_pos):
    print('Drawing...')
    x, y, r = emoji_pos
    top = y - r
    bottom = y + r
    left = x - r
    rigth = x + r

    emoji = cv2.resize(real_emoji, (rigth - left, bottom - top))
    frame[top:bottom, left:rigth] = emoji

    return frame

def detect_emoji(frame):
    circles, mask, _ = emoji_segmentation(frame)
    possible_emojis = get_circle_regions(frame, circles, 1.2)
    
    for emoji in possible_emojis:
        cropped_img, cropped_pos = emoji
        if not is_square(cropped_img): continue
        # Calculate HoG
        # Pass it to trained model
        # if valid, then draw a real emoji
        draw_emoji(frame, EMOJI_1, cropped_pos)

    return frame# * mask

def is_square(img):
    return img.shape[0] == img.shape[1]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Emoji segmentation')
    parser.add_argument('-i', '--image', type=str, action='store', dest='src_img', required=True, help='The input image')
    args = parser.parse_args()

    try:
        EMOJI_1 = cv2.imread('emojis/1.png', cv2.IMREAD_COLOR)

        img = cv2.imread(args.src_img, cv2.IMREAD_COLOR)
        plt_show_img(detect_emoji(img))
        #start_cv_video(0, img_filter=detect_emoji)

    except Exception as error:
        print(error)