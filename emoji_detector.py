from emoji_segmentation import emoji_segmentation, get_circle_regions
from skin_detector.cv_helpers import start_cv_video

def detect_emoji(frame):
    circles, mask = emoji_segmentation(frame)
    possible_emojis = get_circle_regions(frame, circles, 5)

    for emoji in possible_emojis:
        # Calculate HoG
        # Pass it to trained model
        # if valid, then draw a real emoji
        pass

    return mask * frame

if __name__ == '__main__':

    try:
        start_cv_video(0, img_filter=detect_emoji)

    except Exception as error:
        print(error)