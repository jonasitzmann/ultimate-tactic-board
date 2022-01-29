import os
from datetime import datetime
from typing import List
import numpy as np
from numpy import floor, ceil
from sklearn.cluster import KMeans
import digit_classification
import drawer
import cv_utils
import state
import cfg
import cv2

os.system('unset SESSION_MANAGER')


def scan(img_path: str, show_digits=cfg.show_digits, show_circles=cfg.show_circles, labeling_mode=cfg.labeling_mode):
    """
    Extracs the state (poses of players, disc position), from an input image
    :param img_path: path to the input image
    :param show_digits: whether to show intermediate steps of the digit recognition pipeline
    :param show_circles: whether to show intermediate steps of the player localization
    :param labeling_mode: wheter to manually annotate extraced digits to improve future results
    :return: the extracted state
    """
    img = cv2.imread(img_path)
    corners = detect_field(img)
    img = transform_to_birdseye(img, corners)
    gray = cv_utils.min_max_normalize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    ksize = cv_utils.round_to_odd(cfg.resize_factor * cfg.ksize_sharpening)
    gray_blurred = cv2.medianBlur(gray, ksize).astype(np.int32)
    white_emphasized = cv_utils.min_max_normalize(np.clip(gray - gray_blurred, cfg.min_intensity, cfg.max_intensity))
    black_emphasized = cv_utils.min_max_normalize(np.clip(gray_blurred - gray, cfg.min_intensity, cfg.max_intensity))
    binary = cv_utils.adaptive_threshold(white_emphasized, ksize, cfg.offset_binarize_global)
    binary = cv2.medianBlur(binary, cfg.ksize_blur_thresholded)
    binary_black = cv_utils.adaptive_threshold(black_emphasized, ksize, cfg.offset_binarize_global)
    binary_black = cv2.medianBlur(binary_black, cfg.ksize_blur_thresholded)
    ez1, ez2 = find_enzone_lines(img, binary_black)

    annotated, players = img.copy(), []
    for c in find_player_contours(binary):
        players.append(identify_player(img.copy(), c, cfg.radius_pixels, show_digits, labeling_mode))
        annotate_player(annotated, players[-1], c)
    for i, img_step in enumerate([gray, annotated] if show_circles else []):
        cv_utils.display_img(img_step, wait=False, window_name=str(i), pos=i)
    team_1, team_2 = cluster_players_by_color(players)
    # areas = np.array(detect_handdrawings(black_emphasized, circles)) / cfg.resize_factor
    disc_pos = np.array(locate_disc(img, black_emphasized)) / cfg.resize_factor
    areas = None
    return state.State(players_team_1=team_1, players_team_2=team_2, areas=areas, disc=disc_pos)


def find_player_contours(binary, show_circles=False):
    lb, ub = [int(cfg.radius_pixels * factor) for factor in [cfg.player_radius_lb, cfg.player_radius_ub]]
    circles = cv2.HoughCircles(binary, minDist=binary.shape[0]/100, minRadius=lb, maxRadius=ub, **cfg.h_circles_args)[0]
    players_mask = np.zeros_like(binary)
    [cv2.circle(players_mask, (c[0], c[1]), cfg.radius_pixels, cfg.max_intensity, -1) for c in circles.astype(np.uint16)]
    player_contours = cv_utils.find_contours(players_mask)
    for i, img_step in enumerate([binary, players_mask] if show_circles else []):
        cv_utils.display_img(img_step, wait=False, window_name=str(i), pos=i)
    return player_contours


def find_enzone_lines(img, binary_img):
    cv_utils.display_img(binary_img)
    distance_res = 1
    angle_res = np.pi / 180
    threshold = int(img.shape[1] * 0.5)
    endzone_height_px = cfg.endzone_height_m * cfg.resize_factor
    epsilon = int(0.05 * endzone_height_px)
    lines = cv2.HoughLinesP(binary_img, distance_res, angle_res, threshold, minLineLength=threshold, maxLineGap=int(0.2*threshold))
    lines = lines[:, 0]
    for l in lines if lines is not None else []:
        cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    lines = lines[np.abs(lines[:, 1] - lines[:, 3]) < epsilon]
    upper_lines = lines[np.abs(lines[:, 1] - endzone_height_px) < epsilon]
    ul = upper_lines[np.argmax(upper_lines[:, 3] - upper_lines[:, -1])]
    cv2.line(img, (ul[0], ul[1]), (ul[2], ul[3]), (0, 255, 0), 3, cv2.LINE_AA)
    lower_lines = lines[np.abs(lines[:, 1] - img.shape[0] + endzone_height_px) < epsilon]
    ll = lower_lines[np.argmax(lower_lines[:, 3] - lower_lines[:, -1])]
    cv2.line(img, (ll[0], ll[1]), (ll[2], ll[3]), (255, 0, 0), 3, cv2.LINE_AA)
    cv_utils.display_img(img)
    # needed for two reasons:
    # - orientation of playing field (is the red endzone at top or bottom?)
    # - removal of endzone lines for detecting arrows and areas
    return None, None


def transform_to_birdseye(img, corners):
    corners = sort_vertices_clockwise(corners)
    l1, l2 = np.linalg.norm(corners[:2] - corners[1:3], 2, axis=1)
    if l2 > l1:
        corners = corners[[1, 2, 3, 0]]  # rotate 90°
    new_h, new_w = cfg.field_height_m * cfg.resize_factor, cfg.field_width_m * cfg.resize_factor
    corners_out = np.array([(0, 0), (0, new_h), (new_w, new_h), (new_w, 0)], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, corners_out)
    return cv2.warpPerspective(img, transform, (new_w, new_h))


def sort_vertices_clockwise(vertices):
    center = np.mean(vertices, dtype=np.int16, axis=0)
    angles = [np.arctan2(*(corner - center)) for corner in vertices]
    return np.array([corner for angle, corner in sorted(zip(angles, vertices))], np.float32)


def locate_disc(img: np.array, gray_sharp) -> np.array:
    """
    :param img: an image containing the (transformed) tactics board
    :return: disc coordinates in the image img
    """
    saturation = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    binary = cv_utils.adaptive_threshold(gray_sharp, cv_utils.round_to_odd(cfg.ksize_sharpening + cfg.resize_factor), cfg.offset_binarize_global)
    binary = cv2.medianBlur(binary, cfg.ksize_blur_thresholded)
    lb, ub = [int(cfg.radius_pixels_disc * factor) for factor in [cfg.player_radius_lb, cfg.player_radius_ub]]
    circles = cv2.HoughCircles(binary, minDist=img.shape[0]/100, minRadius=lb, maxRadius=ub, **cfg.h_circles_args_disc)[0]
    disc_mask = np.zeros_like(binary)
    best_saturation, best_position = cfg.min_intensity, None
    for x, y, radius in circles.astype(np.uint16):
        cv2.circle(disc_mask, (x, y), cfg.radius_pixels_disc, cfg.max_intensity, -1)
        current_saturation = saturation[y, x]
        if current_saturation > best_saturation:
            best_saturation, best_position = current_saturation, [x, y]
    disc_pos = np.array(best_position)
    return disc_pos


def detect_handdrawings(gray_img, player_contours, show_intermediate_results=True):
    cv_utils.display_img(gray_img, wait=False) if show_intermediate_results else None
    # todo: this is an experiment and has only been testen on a single image
    gray_img[:, -5:] = gray_img[:, -6, None]  # todo: hack to remove artifact on right border
    edges = cv2.Canny(gray_img, 100, 200)
    [cv2.circle(edges, (c[0], c[1]), int(c[2]*2.2), cfg.min_intensity, -1) for c in player_contours.astype(np.uint16)]
    # remove endzone lines
    lw = int(0.5 * cfg.resize_factor)  # 1.5m in pixels
    for height in [cfg.endzone_height_m, cfg.field_height_m - cfg.endzone_height_m]:
        pts = np.array([[0, height], [cfg.field_width_m, height]])
        pts = (pts * cfg.resize_factor).astype(np.int32)
        cv2.line(edges, *pts, cfg.min_intensity, lw)  # use other method to remove the endzone line
    cv_utils.display_img(edges, wait=False) if show_intermediate_results else None
    lw = cv_utils.round_to_odd(lw)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lw, lw))
    iters = 3
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, None, None, iters, cv2.BORDER_CONSTANT, cfg.min_intensity)
    cv_utils.display_img(edges, wait=False) if show_intermediate_results else None
    contours = cv_utils.find_contours(edges)
    areas = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        poly = cv2.approxPolyDP(hull, 100, True)  # todo: how to configure the allowed error?
        hull_area, c_area = [cv2.contourArea(c) for c in [hull, contour]]
        if c_area > 100:
            if hull_area / c_area < 1.2:
                poly = [np.array(p[0]) for p in poly]
                areas.append(np.array(poly))

    cv2.drawContours(gray_img, areas, -1, cfg.max_intensity, -1)
    cv_utils.display_img(gray_img, wait=False) if show_intermediate_results else None
    areas = np.array(areas)
    return areas


def annotate_player(img, player, player_contour):
    """
    Draws the detected player and its label on a given image
    :param img: image to draw the player on
    :param player: player to be drawn
    :param player_contour: detected contour of the player
    """
    cv2.drawContours(img, [player_contour], 0, player.color.astype(np.uint8).tolist(), -1)
    text_pos = (player.pos * cfg.resize_factor + np.array([-17, 17])).astype(np.int32)
    cv2.putText(img, player.label, text_pos, 1, cfg.font_size, (0, 0, 0), 3)


def cluster_players_by_color(players: List[state.Player]):
    """
    clusters players based on the background color.
    Uses k-means with two centroids
    :param players: list of players to be clustered
    :return: two lists containing the players of team 1 (light) and team 2 (dark)
    """
    team_1, team_2 = [], []
    kmeans = KMeans(n_clusters=2)
    colors = np.array([p.color for p in players])
    kmeans.fit(np.array(colors))
    kmeans.cluster_centers_ = np.array(sorted(kmeans.cluster_centers_, key=lambda x: sum(x**2)))
    preds = kmeans.predict(np.array(colors))
    for pred, player in zip(preds, players):
        (team_1 if pred else team_2).append(player)
    return team_1, team_2


def detect_field(img: np.ndarray, show_edges=cfg.show_edges) -> np.ndarray:
    """
    detects an ultimate field on a given input image
    :param img: numpy array representing the input image
    :param show_edges: wheter to show the (intermediate) binarized image
    :return: corners of the field
    """
    img_gray = cv_utils.min_max_normalize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    img_gray = cv2.medianBlur(img_gray, cfg.ksize_initial_blur)
    edges = cv_utils.adaptive_threshold(img_gray, cfg.ksize_thresh_field, cfg.offset_thresh_field)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    corners = max_area = None
    for c in contours:
        area = cv2.contourArea(c)
        if (max_area is None) or (area > max_area):
            hull_candidate = cv2.approxPolyDP(cv2.convexHull(c), cfg.field_detection_poly_epsilon, True)
            if len(hull_candidate) == 4:
                corners, max_area = hull_candidate[:, 0], area
    if show_edges:
        cv_utils.display_img(edges, wait=False)
        cv2.drawContours(edges, [corners], -1, cfg.medium_intensity, cfg.resize_factor // 2)
        cv_utils.display_img(edges, window_name='edges', wait=True)
    return corners


def identify_player(img, contour, radius_pixels, show_digits, labeling_mode=False) -> state.Player:
    """
    recognizes featues of a player (as in state.Player) with a given contour in a given image
    :param img: image to recognize the player in
    :param contour: contour of the player in the image img
    :param radius_pixels: approximate radius of the player
    :param show_digits: whether to show images of intermediate steps
    :param labeling_mode: wheter to manually annotate extraced digits to improve future results
    :return: state.Player object representing the player
    """
    cropped_contour = np.zeros_like(img[:, :, 0])
    pos = cv_utils.get_contour_center(contour)
    cv2.drawContours(cropped_contour, [contour], 0, cfg.max_intensity, cfg.filled)
    crop = cv_utils.crop_to_content(cropped_contour, img, radius_pixels // 2)
    crop_binary, digit_hull = extract_digit(crop)
    angle, frame_contour = estimate_frame(crop, digit_hull, show_digits)
    crop_rotated = rotate_to_mnist(crop_binary, -angle)
    classification = digit_classification.classify_img_by_examples(crop_rotated, show=show_digits)
    if show_digits:
        for i, img in enumerate([crop_binary, crop_rotated]):
            cv_utils.display_img(img, str(i + 3), False, i + 3)
        cv2.waitKey(0)
    background_mask = cv_utils.crop_to_content(cropped_contour, cropped_contour, radius_pixels // 2)
    for c in [frame_contour, digit_hull]:  # digit_contour would be more accurate than digit_hull
        [cv2.drawContours(background_mask, [c], 0, 0, lw) for lw in [-1, 2]]
    background_color = np.median(crop[background_mask > 0].reshape(-1, 3), axis=0).astype(np.uint8)
    player = state.Player(pos / cfg.resize_factor, angle, str(classification), background_color)
    label_refenrence_player(crop_rotated) if labeling_mode else None
    return player


def estimate_frame(crop, digit_hull, show_digits):
    # todo: avoid cropping with margin
    # todo: too many dependencies
    """
    estimates the orientation and convex hull of a player's frame in a given image
    :param crop: small image to recognize the player in
    :param digit_hull: convex hull of the digit (should be contained in contour)
    :param show_digits: whether to show images of intermediate steps
    :return: orientation and convex hull of the player's frame
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, cfg.ksize_blur_crop).astype(np.int32)
    gray_sharp = np.clip(gray - gray_blurred, cfg.min_intensity, cfg.max_intensity)
    cv2.drawContours(gray_sharp, [digit_hull], -1, cfg.min_intensity, -1)  # remove digit
    gray_sharp = cv_utils.min_max_normalize(gray_sharp)
    frame = cv_utils.adaptive_threshold(gray_sharp, cfg.ksize_thresh_frame, cfg.offset_thresh_frame)
    contours = cv_utils.find_contours(frame)
    frame_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    frame_only = np.zeros_like(frame)
    cv2.drawContours(frame_only, [frame_contour], -1, cfg.max_intensity, -1)
    thinned = cv2.ximgproc.thinning(frame_only)  # install opencv-contrib-python!
    thinned_pts = np.array(np.where(thinned == cfg.max_intensity)[::-1]).T
    thinned_hull = cv2.convexHull(thinned_pts)
    frame_2 = np.zeros_like(frame)
    cv2.drawContours(frame_2, [thinned_hull], -1, cfg.max_intensity, 1)
    cv2.drawContours(frame_2, [frame_contour], 0, cfg.medium_intensity, -1)
    opening_point = np.array(np.where(frame_2 == cfg.max_intensity)).mean(axis=1)[::-1]
    center = cv_utils.get_contour_center(cv2.convexHull(frame_contour))
    orientation = (np.rad2deg(np.arctan2(*(opening_point - center))) + 180) % 360
    cv2.line(frame_2, center.astype(np.uint8), opening_point.astype(np.uint8), cfg.medium_intensity, 1)
    if show_digits:
        for i, img in enumerate([gray_sharp, frame, frame_2]):
            cv_utils.display_img(img, str(i), False, i)
        cv2.waitKey(100)
    return orientation, frame_contour


def rotate_to_mnist(img, angle):
    """
    rotates an image and resizes its content to the MNIST format (28 x 28 pixels)
    :param img: image to be rotated
    :param angle: rotation angle (clockwise)
    :return: rotated image in mnist format
    """
    padded = np.pad(img, max(img.shape[:2]))
    rotated = cv_utils.crop_to_content(cv_utils.rotate_img(padded, angle))
    longer_side = max(rotated.shape[:2])
    scale = (cfg.digit_target_size - 6) / longer_side
    new_size = (np.array(rotated.shape[:2]) * scale).astype(np.int32)[::-1]
    rotated = cv2.resize(rotated, new_size)
    pad_lr, pad_tb = (cfg.digit_target_size - new_size) / 2
    pad_sizes = np.array([ceil(pad_tb), floor(pad_tb), ceil(pad_lr), floor(pad_lr)], dtype=np.int32)
    rotated = cv2.copyMakeBorder(rotated, *pad_sizes, cv2.BORDER_CONSTANT)
    return rotated


def extract_digit(img):
    """
    finds a digit in an image
    :param img: the image containg the digit
    :return: a binary crop of the digit and the convex hull of the digit in img
    """
    gray = cv_utils.min_max_normalize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    crop_binary = cv_utils.adaptive_threshold(gray, cfg.ksize_thresh_digits, cfg.offset_thresh_digits)
    contours = cv_utils.find_contours(crop_binary)
    best_hull = min_area = None
    center = tuple(int(x) for x in (np.array(img.shape[:2]) / 2))
    for c in contours:
        single_contour_mask, hull = np.zeros_like(crop_binary), cv2.convexHull(c)
        if cv2.pointPolygonTest(hull, center, True) > -5:
            if cv2.contourArea(hull) > cfg.min_contour_area_digit:
                cv2.drawContours(single_contour_mask, [hull], -1, cfg.max_intensity, -1)
                is_white = np.median(crop_binary[single_contour_mask == cfg.max_intensity]) == cfg.max_intensity
                if is_white:
                    area = cv2.contourArea(hull)
                    if min_area is None or area < min_area:
                        min_area, best_hull = area, hull
    if best_hull is None:
        cv2.drawContours(img, contours, -1, (cfg.max_intensity, cfg.medium_intensity, cfg.min_intensity), 1)
        cv_utils.display_imgs([img, crop_binary])
    digit_hull_mask = np.zeros_like(img[:, :, 0])
    cv2.drawContours(digit_hull_mask, [best_hull], 0, cfg.max_intensity, -1)
    crop_binary[digit_hull_mask == cfg.min_intensity] = cfg.min_intensity
    crop_binary = cv_utils.crop_to_content(crop_binary, crop_binary)
    return crop_binary, best_hull


def label_refenrence_player(img):
    """
    asks the user to input the player label (1-7) and saves label + image to recognize it in the future
    :param img: MNIST-like image
    """
    cv_utils.display_img(img, 'digit', wait=False, pos=1)
    cv2.waitKey(100)
    label = input('which digit is this?\n')
    dirname = f'{cfg.reference_digits_dir}/{label}'
    os.makedirs(dirname, exist_ok=True)
    timestamp = datetime.now().strftime("%d-%b-%Y__%H-%M-%S")
    filename = f'{dirname}/{timestamp}.png'
    cv2.imwrite(filename, img)


if __name__ == '__main__':
    state = scan(cfg.input_imgs_dir + 'ho-stack-1.jpg')
    surface = drawer.draw_scene(state)
    drawer.show(surface, wait=0)