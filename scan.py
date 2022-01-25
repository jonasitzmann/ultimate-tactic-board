from datetime import datetime
from typing import List
import cv2  # install opencv-contrib-python
import os
import numpy as np
from numpy import floor, ceil
from sklearn.cluster import KMeans
import digit_classification
import cfg
import state
import drawer

os.system('unset SESSION_MANAGER')
show_count = 0


def scan(img_path, show_digits=False, show_circles=False, record_examples=False):
    img = cv2.imread(img_path)
    new_h, new_w = cfg.field_height_m * cfg.resize_factor, cfg.field_width_m * cfg.resize_factor
    sorted_corners = detect_field(img)
    source_points = np.array(sorted_corners, dtype=np.float32)
    target_points = np.array([(new_w, new_h), (new_w, 0), (0, 0), (0, new_h)], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source_points, target_points[[1, 2, 3, 0]])
    img_transformed = cv2.warpPerspective(img, transform, (new_w, new_h))
    gray = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2GRAY)
    gray = (gray.astype(np.float32) * cfg.max_intensity / gray.max()).astype(np.uint8)
    ksize = round_to_odd(cfg.resize_factor * cfg.ksize_sharpening)
    gray_blurred = cv2.medianBlur(gray, ksize)
    gray_sharp = np.clip(gray.astype(np.float32) - gray_blurred.astype(np.int32), cfg.min_intensity, cfg.max_intensity)
    gray_sharp = min_max_normalize(gray_sharp)
    thresholded = adaptive_threshold(gray_sharp, ksize, -5)
    thresholded = cv2.medianBlur(thresholded, cfg.ksize_blur_thresholded)
    radius_pixels = cfg.radius_players_cm * new_h // cfg.field_height_m
    circles = cv2.HoughCircles(
        thresholded,
        cv2.HOUGH_GRADIENT,
        1.8, new_h / 100,
        param1=150,
        param2=30,
        minRadius=int(radius_pixels * cfg.player_radius_lb),
        maxRadius=int(radius_pixels * cfg.player_radius_ub)
    )[0].astype(np.uint16)
    mask = np.zeros_like(thresholded)
    [cv2.circle(mask, (c[0], c[1]), int(radius_pixels*0.95), cfg.max_intensity, -1) for c in circles]
    annotated = img_transformed.copy()
    players = []
    for c in findContours(mask):
        crop, player = process_player(img_transformed.copy(), c, radius_pixels, show_digits)
        players.append(player)
        save_example_image(crop) if record_examples else None
        cv2.drawContours(annotated, [c], 0, player.color.astype(np.uint8).tolist(), -1)
        text_pos = (player.pos * cfg.resize_factor + np.array([-17, 17])).astype(np.int32)
        cv2.putText(annotated, player.label, text_pos, 1, cfg.font_size, (0, 0, 0), 3)
    for img in [img_transformed, thresholded, mask, annotated] if show_circles else []:
        display_img(img, wait=False)
    players_team_1, player_team_2 = cluster_players_by_color(players)
    return state.State(players_team_1=players_team_1, players_team_2=player_team_2)


def findContours(img):
    return cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]


def round_to_odd(x):
    """
    :param x: some number (float or int)
    :return: nearst odd integer
    """
    return round((x - 1) / 2) * 2 + 1


def cluster_players_by_color(players: List[state.Player]):
    players_team_1, players_team_2 = [], []
    kmeans = KMeans(n_clusters=2)
    colors = np.array([p.color for p in players])
    kmeans.fit(np.array(colors))
    kmeans.cluster_centers_ = np.array(sorted(kmeans.cluster_centers_, key=lambda x: sum(x**2)))  # sort by darkness
    preds = kmeans.predict(np.array(colors))
    for pred, player in zip(preds, players):
        (players_team_1 if pred else players_team_2).append(player)
    return players_team_1, players_team_2


def adaptive_threshold(img, ksize, c):
    return cv2.adaptiveThreshold(img, cfg.max_intensity, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ksize, c)


def detect_field(img, show_edges=False):
    img_gray = min_max_normalize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    img_gray = cv2.medianBlur(img_gray, cfg.ksize_initial_blur)
    edges = adaptive_threshold(img_gray, 15, 1)
    display_img(edges, window_name='edges') if show_edges else None
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    corners = max_area = None
    for c in contours:
        area = cv2.contourArea(c)
        if (max_area is None) or (area > max_area):
            hull_candidate = cv2.approxPolyDP(cv2.convexHull(c), cfg.field_detection_poly_epsilon, True)
            if len(hull_candidate) == 4:
                corners = hull_candidate[:, 0]
                max_area = area
    center = np.mean(corners, dtype=np.int16, axis=0)
    angles = [np.arctan2(*(corner - center)) for corner in corners]  # todo use fiducials to identify corners
    sorted_corners = np.array([corner for angle, corner in sorted(zip(angles, corners))])  # clockwise from tl
    return sorted_corners


def process_player(image, contour, radius_pixels, show_digits):
    mask = np.zeros_like(image[:, :, 0])
    cv2.drawContours(mask, [contour], 0, 255, -1)
    crop = crop_to_content(mask, image, radius_pixels // 2)
    crop_binary, best_hull = extract_digit(crop)
    center, angle, frame_contour = find_rotation(image, contour, mask, best_hull, radius_pixels, show_digits)
    crop_rotated = rotate(crop_binary, -angle)
    classification = digit_classification.classify_image_by_examples(crop_rotated, show=show_digits)
    if show_digits:
        for i, img in enumerate([crop_binary, crop_rotated]):
            display_img(img, 70, str(i+3), False, i+3)
        cv2.waitKey(0)
    background_mask = crop_to_content(mask, mask, radius_pixels // 2)
    for detected_contour in [frame_contour, best_hull]:  # actually digit_contour would be more accurate than best_hull
        for line_width in -1, 2:
            cv2.drawContours(background_mask, [detected_contour], 0, 0, line_width)
    background_pixels = crop[background_mask > 0]
    background_pixels = background_pixels.reshape(-1, 3)
    background_color = np.median(background_pixels, axis=0).astype(np.uint8)
    pos = np.mean(contour, axis=0)[0] / cfg.resize_factor
    player = state.Player(pos, angle, str(classification), background_color)
    return crop_rotated, player


def find_rotation(image, contour, mask, best_hull, radius_pixels, show_digits):
    frame_mask = np.zeros_like(image[:, :, 0])
    cv2.drawContours(frame_mask, [contour], 0, 255, 9)
    gray = crop_to_content(mask, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), radius_pixels // 2)
    ksize = 11
    gray_blurred = cv2.medianBlur(gray, ksize)
    gray_sharp = np.clip(gray.astype(np.float32) - gray_blurred.astype(np.int32), 0, 255)
    cv2.drawContours(gray_sharp, [best_hull], -1, 0, -1)
    gray_sharp = min_max_normalize(gray_sharp)
    frame = adaptive_threshold(gray_sharp, 45, -10)
    frame_mask = crop_to_content(mask, frame_mask, radius_pixels // 2)
    frame[frame_mask == 0] = 0
    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, 200, 1)
    frame = cv2.dilate(frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    frame_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    frame_only = np.zeros_like(frame)
    cv2.drawContours(frame_only, [frame_contour], -1, 255, -1)
    thinned = cv2.ximgproc.thinning(frame_only)  # install opencv-contrib-python!
    thinned_pts = np.array(np.where(thinned == 255)[::-1]).T
    thinned_hull = cv2.convexHull(thinned_pts)

    frame_2 = np.zeros_like(frame)
    cv2.drawContours(frame_2, [thinned_hull], -1, 255, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    frame_2 = cv2.erode(frame_2, kernel)
    cv2.drawContours(frame_2, [frame_contour], 0, 150, -1)
    opening_point = np.array(np.where(frame_2 == 255)).mean(axis=1)[::-1]

    M = cv2.moments(cv2.convexHull(frame_contour))
    center = np.array([M['m10']/M['m00'], M['m01']/M['m00']])
    angle = np.arctan2(*(opening_point - center))
    angle = (angle / np.pi * 180 + 180) % 360

    cv2.line(frame_2, center.astype(np.uint8), opening_point.astype(np.uint8), 150, 1)
    if show_digits:
        for i, img in enumerate([gray_sharp, thinned, frame_2]):
            display_img(img, 70, str(i), False, i)
        cv2.waitKey(100)
    return center, angle, frame_contour


def rotate(crop_binary, angle):
    longer_side = max(crop_binary.shape[:2])
    crop_padded = np.pad(crop_binary, longer_side)
    crop_rotated = rotate_image(crop_padded, angle)
    crop_rotated = crop_to_content(crop_rotated, crop_rotated)
    longer_side = max(crop_rotated.shape[:2])
    crop_scale = (cfg.digit_target_size - 6) / longer_side
    new_size = (np.array(crop_rotated.shape[:2]) * crop_scale).astype(np.int32)
    new_size = np.array([new_size[1], new_size[0]])
    crop_rotated = cv2.resize(crop_rotated, new_size)
    pad_lr, pad_tb = (cfg.digit_target_size - new_size) / 2
    pad_sizes = np.array([ceil(pad_tb), floor(pad_tb), ceil(pad_lr), floor(pad_lr)], dtype=np.int32)
    crop_rotated = cv2.copyMakeBorder(crop_rotated, *pad_sizes, cv2.BORDER_CONSTANT)
    return crop_rotated


def extract_digit(crop):
    gray = min_max_normalize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    crop_binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -6)
    contours, _ = cv2.findContours(crop_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    best_hull = min_area = None
    center = (np.array(crop.shape[:2]) / 2).astype(np.int32)
    center = tuple(int(x) for x in center)
    for c in contours:
        single_contour_mask = np.zeros_like(crop_binary)
        hull = cv2.convexHull(c)
        if cv2.pointPolygonTest(hull, center, True) > -5:
            if cv2.contourArea(hull) > 20:
                cv2.drawContours(single_contour_mask, [hull], -1, 255, -1)
                is_white = np.median(crop_binary[single_contour_mask == 255]) == 255
                if is_white:
                    area = cv2.contourArea(hull)
                    if min_area is None or area < min_area:
                        min_area = area
                        best_hull = hull
    if best_hull is None:
        cv2.drawContours(crop, contours, -1, (255, 100, 0), 1)
        display_img(crop, 100, wait=False)
        display_img(crop_binary, 100)
    mask_2 = np.zeros_like(crop[:, :, 0])
    cv2.drawContours(mask_2, [best_hull], 0, 255, -1)
    crop_binary[mask_2 == 0] = 0
    crop_binary = crop_to_content(crop_binary, crop_binary)
    return crop_binary, best_hull


def min_max_normalize(img):
    normalized = img.copy().astype(np.float32)
    normalized += cfg.min_intensity - img.min()
    normalized *= cfg.max_intensity / img.max()
    return normalized.astype(np.uint8)


def save_example_image(crop_rotated):
    display_img(crop_rotated, 100, 'digit', wait=False, pos=1)
    cv2.waitKey(100)
    label = input('which digit is this?\n')
    dirname = f'{cfg.reference_digits_dir}/{label}'
    os.makedirs(dirname, exist_ok=True)
    timestamp = datetime.now().strftime("%d-%b-%Y__%H-%M-%S")
    filename = f'{dirname}/{timestamp}.png'
    cv2.imwrite(filename, crop_rotated)


def crop_to_content(mask, image_to_crop, margin=0):
    (y, x) = np.where(mask)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    topx = max(topx - margin, 0)
    topy = max(topy - margin, 0)
    bottomx = min(bottomx + margin, image_to_crop.shape[1])
    bottomy = min(bottomy + margin, image_to_crop.shape[0])
    crop = image_to_crop[topy: bottomy+1, topx:bottomx+1]
    return crop


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def display_img(img, scale=8, window_name=None, wait=True, pos=None):
    global show_count
    scale /= cfg.resize_factor
    pos = show_count if pos is None else pos
    h, w = (np.array(img.shape[:2]) * scale).astype(np.int32)
    window_name = f'{show_count}' if window_name is None else window_name
    cv2.imshow(window_name, cv2.resize(img, (w, h)))
    window_x = (cfg.resolution_x // cfg.max_num_windows) * (pos % cfg.max_num_windows)
    cv2.moveWindow(window_name, window_x, 0)
    show_count += 1
    cv2.waitKey(0) if wait else None


if __name__ == '__main__':
    state = scan(cfg.input_images_dir + 'disc.jpg', show_digits=False, show_circles=True)
    surface = drawer.draw_scene(state)
    drawer.show(surface, wait=0)
