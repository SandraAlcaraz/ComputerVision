import sys
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def plt_hist(data, title='', bins=256):
    plt.hist(data, bins)
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

def plt_show_img(img, title=''):
    plt.imshow(img, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

def cv2_show_img(img):
    ANY_KEY = 0
    cv2.imshow('', img)
    cv2.waitKey(ANY_KEY)

def show_compared_imgs(*imgs, title=''):
    img_comparison = np.concatenate((imgs[0], imgs[1]), axis=1)

    if len(imgs) > 2:
        for i in range(2, len(imgs) - 1):
            img_comparison = np.concatenate((img_comparison, imgs[i]), axis=1)
    
    plt_show_img(img_comparison, title)

def start_video(camera = 0, img_filter = None, *img_filter_params):
    cap = cv2.VideoCapture(camera)
    plt.xticks([])
    plt.yticks([])
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_view = plt.imshow(frame, cmap='gray')

    def update(i):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if img_filter is not None:
            frame = img_filter(frame, *img_filter_params)

        frame_view.set_data(frame)

    _ = FuncAnimation(plt.gcf(), update, interval=200)
    plt.show()

def start_cv_video(camera = 0, img_filter = None, *img_filter_params):
    cap = cv2.VideoCapture(camera)
    print('Press q to exit...')
    while(True):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if img_filter is not None:
            frame = img_filter(frame, *img_filter_params)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
