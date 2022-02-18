import cv2
import cfg
import numpy as np
show_count = 0


def min_max_normalize(img):
    normalized = img.copy().astype(np.float32)
    normalized += cfg.min_intensity - img.min()
    normalized *= cfg.max_intensity / img.max()
    return normalized.astype(np.uint8)


def crop_to_content(mask, img_to_crop=None, margin=0):
    img_to_crop = mask if img_to_crop is None else img_to_crop
    (y, x) = np.where(mask)
    if len(y) == 0:
        return None
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    topx = max(topx - margin, 0)
    topy = max(topy - margin, 0)
    bottomx = min(bottomx + margin, img_to_crop.shape[1])
    bottomy = min(bottomy + margin, img_to_crop.shape[0])
    crop = img_to_crop[topy: bottomy+1, topx:bottomx+1]
    return crop


def rotate_img(img, angle):
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


def display_img(img, window_name=None, wait=True, pos=None, scale=None):
    global show_count
    if scale is None:
        border_px = 10
        scale = (cfg.resolution_x / cfg.max_num_windows - 2 * border_px) / img.shape[1]
    else:
        scale /= cfg.resize_factor
    pos = show_count if pos is None else pos
    h, w = (np.array(img.shape[:2]) * scale).astype(np.int32)
    window_name = f'{show_count}' if window_name is None else window_name
    cv2.imshow(window_name, cv2.resize(img, (w, h)))
    window_x = (cfg.resolution_x // cfg.max_num_windows) * (pos % cfg.max_num_windows)
    cv2.moveWindow(window_name, window_x, 0)
    show_count += 1
    cv2.waitKey(0) if wait else None


def display_imgs(imgs, start_pos=0, scale=None):
    for i, img in enumerate(imgs):
        display_img(img, str(i), False, i + start_pos, scale)
    cv2.waitKey(0)


def get_contour_center(contour):
    moments = cv2.moments(contour)
    return np.array([moments['m10']/moments['m00'], moments['m01']/moments['m00']])


def find_contours(img):
    """
    wrapper for cv2.findContours
    """
    return cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]


def round_to_odd(x):
    """
    :param x: some number (float or int)
    :return: nearst odd integer
    """
    return int(round((x - 1) / 2) * 2 + 1)

def round_to_even(x):
    """
    :param x: some number (float or int)
    :return: nearst odd integer
    """
    return int(round(x / 2) * 2)

def adaptive_threshold(img, ksize, c):
    """
    wrapper for cv2.adaptiveThreshold
    """
    return cv2.adaptiveThreshold(img, cfg.max_intensity, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ksize, c)

