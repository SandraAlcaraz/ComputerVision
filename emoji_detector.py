from emoji_segmentation import emoji_segmentation, get_circle_regions
from skin_detector.cv_helpers import plt_show_img, cv2_show_img

from cv2 import cv2
import numpy as np
#import keras

EMOJI_DICT = {}
#EMOJI_MODEL = keras.models.load_model('filter_model.h5')
#MOUTH_MODEL = keras.models.load_model('filter_model_complete.h5')
MASK_EMOJIS = None
BACK_EMOJIS = None

def classify_emoji(img):
    return 1
    res = EMOJI_MODEL.predict(np.array([img]) / 255 )[0]
    return 1 if(res[1] > res[0]) else 0

def classify_mouth(img):
    return 1
    images = extract_face_features(img)
    max_score = 0
    max_index = 0
    for img in images:
        img = resize_image(img, 100)
        res = MOUTH_MODEL.predict(np.array([img])/255)[0]
        mx = max(res)
        i = res.tolist().index(mx)
        if i > 0 and mx > max_score:
            max_score = mx
            max_index = i
    return max_index

def extract_face_features(img):
    def has_more_than_3_elements(cnt_list):
        return len(cnt_list) > 2

    original = img.copy()
    img = process_emoji(img)
    
    filtered_cnt, max_cnt = get_contours(img)
    if max_cnt is None or len(filtered_cnt) == 0: return [] 
    new_filter = get_inside_face_cnt(filtered_cnt, max_cnt)
    if not has_more_than_3_elements(new_filter): return []
    emoji_fts = []

    for mouth_emoji in new_filter:
        if mouth_emoji is None: continue
        M = cv2.moments(mouth_emoji)
        cy = int(M['m01']/M['m00'])
        x,y,w,h = cv2.boundingRect(mouth_emoji)
        roi = original[y:y+h, x:x+w].copy()
        emoji_fts.append(roi)
    
    return emoji_fts

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
    real_emoji, inverse_mask = EMOJI_DICT.get(emoji_index)
    x, y, r = emoji_pos
    top = y - r
    bottom = y + r
    left = x - r
    rigth = x + r

    emoji = cv2.resize(real_emoji, (rigth - left, bottom - top))
    inverse_mask = cv2.resize(inverse_mask, (rigth - left, bottom - top))

    MASK_EMOJIS[top:bottom, left:rigth] = cv2.bitwise_not(inverse_mask)
    BACK_EMOJIS[top:bottom, left:rigth] = emoji

    overlap_area = frame[top:bottom, left:rigth]
    overlap_area = cv2.bitwise_and(overlap_area, overlap_area, mask=inverse_mask)
    overlap_area = cv2.add(overlap_area, emoji)
    frame[top:bottom, left:rigth] = overlap_area

    return frame

possible_emojis = []
def detect_emoji(frame):
    global possible_emojis
    global MASK_EMOJIS, BACK_EMOJIS
    
    MASK_EMOJIS = np.zeros((frame.shape[0], frame.shape[1]), dtype='uint8')
    BACK_EMOJIS = np.zeros((frame.shape[0], frame.shape[1], 3), dtype='uint8')
    print('recalculating')
    circles, mask, _ = emoji_segmentation(frame)
    possible_emojis = get_circle_regions(frame, circles, 1.2)
    
    # true_emojis = []
    # for emoji_p in possible_emojis:
    #     _, (x, y, r) = emoji_p
    #     for emoji_c in possible_emojis:
    #         _, (x_c, y_c, r_c) = emoji_c
    #         d = ((x-x_c)**2 + (y - y_c)**2)**0.5
            
    #         if r + r_c > d:
    #             print('holiwis')
    #             if r > r_c:
    #                 # Sabemos que r_c está dentro
    #                 true_emojis.append(emoji_p)
    #             else:
    #                 true_emojis.append(emoji_c)
                    
    
    for emoji in possible_emojis:
        if emoji is None: continue
        cropped_img, cropped_pos = emoji
        if cropped_img is None: continue
        if not is_square(cropped_img): continue
        cropped_img = resize_image(cropped_img)
        is_emoji = classify_emoji(cropped_img)
        if not is_emoji: continue
        emoji_type = classify_mouth(cropped_img)
        if not emoji_type: continue
        draw_emoji(frame, emoji_type, cropped_pos)

    return BACK_EMOJIS, MASK_EMOJIS#frame# * mask

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

def read_video(video='video.MOV'):
    cap = cv2.VideoCapture(video)
    if (not cap.isOpened()): 
        print("Error opening video stream or file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            # cv2.imshow('', frame)
            frame = detect_emoji(frame)
            out.write(frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def start_cv_video(camera = 0, img_filter = None):
    cap = cv2.VideoCapture(camera)
    print('Press q to exit...')
    counter = 0
    mask = None
    emojis = None
    _, frame = cap.read()
    emojis, mask = img_filter(frame)
    while(True):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if img_filter is not None:
            if counter == 0:
                emojis, mask = img_filter(frame)

            if emojis is not None and mask is not None:
                frame = cv2.bitwise_and(frame, frame, mask=mask)
                frame = cv2.add(frame, emojis)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('', frame)
        print(counter)
        counter += 1
        counter = counter % 3
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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
        emojis, mask = detect_emoji(img)
        mask = cv2.bitwise_not(mask)
        img = cv2.bitwise_and(img, img, mask=mask)
        img = cv2.add(img, emojis)
        cv2_show_img(img)
        # plt_show_img(detect_emoji(img))
        #start_cv_video(1, img_filter=detect_emoji)

    except Exception as error:
        print(error)