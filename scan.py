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


def get_median_line_color_in_img(line, img, show_median_color=cfg.show_median_color):
    mask = np.zeros_like(img[:, :, 0])
    cv2.line(mask, line[0], line[1], cfg.max_intensity, 1)
    median_color = np.median(img[mask == cfg.max_intensity], axis=0)
    if show_median_color:
        colored_img = (np.ones((100, 100, 3)) * median_color).astype(np.uint8)
        cv_utils.display_img(colored_img)
    return median_color


def rotate_red_ez_to_top_inplace(ez_lines, img, *imgs):
    (b1, g1, r1), (b2, g2, r2) = [get_median_line_color_in_img(line, img) for line in ez_lines]
    if r1 - b1 < r2 - b2:  # blue endzone is on top currently
        for img_to_rotate in [img, *imgs]:
            img_to_rotate[:] = np.rot90(np.rot90(img_to_rotate))


def draw_field(img, ez_lines, show=cfg.show_field):
    line_width_m = 0.4
    lw_px = cfg.resize_factor * line_width_m
    lw_px, half_lw_px = int(lw_px), int(lw_px / 2)
    x0 = y0 = half_lw_px
    y1, x1 = np.array(img.shape[:2]) - half_lw_px
    field = np.zeros_like(img[:, :, 0])
    lines = [((x0, y0), (x0, y1)), ((x1, y0), (x1, y1)), ((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)), *ez_lines]
    [cv2.line(field, p1, p2, cfg.max_intensity, lw_px) for p1, p2 in lines]
    cv_utils.display_img(field) if show else None
    return field



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
    ez_lines = find_enzone_lines(img, binary_black)
    field = draw_field(img, ez_lines)
    rotate_red_ez_to_top_inplace(ez_lines, img, gray, gray_blurred, white_emphasized, black_emphasized, binary, binary_black, field)
    annotated, players = img.copy(), []
    player_contours, circles = find_player_contours(binary)
    for c in player_contours:
        players.append(identify_player(img.copy(), c, cfg.radius_pixels, show_digits, labeling_mode))
        annotate_player(annotated, players[-1], c)
    team_1, team_2 = cluster_players_by_color(players)
    disc_pos = np.array(locate_disc(img, black_emphasized))
    areas = np.array(detect_handdrawings(black_emphasized, circles, field, disc_pos)) / cfg.resize_factor
    for i, img_step in enumerate([gray, annotated] if show_circles else []):
        cv_utils.display_img(img_step, wait=False, window_name=str(i), pos=i)

    return state.State(players_team_1=team_1, players_team_2=team_2, areas=areas, disc=disc_pos / cfg.resize_factor)


def find_player_contours(binary, show_circles=False):
    lb, ub = [int(cfg.radius_pixels * factor) for factor in [cfg.player_radius_lb, cfg.player_radius_ub]]
    circles = cv2.HoughCircles(binary, minDist=binary.shape[0]/100, minRadius=lb, maxRadius=ub, **cfg.h_circles_args)[0].astype(np.uint16)
    players_mask = np.zeros_like(binary)
    [cv2.circle(players_mask, (c[0], c[1]), cfg.radius_pixels, cfg.max_intensity, -1) for c in circles]
    player_contours = cv_utils.find_contours(players_mask)
    for i, img_step in enumerate([binary, players_mask] if show_circles else []):
        cv_utils.display_img(img_step, wait=False, window_name=str(i), pos=i)
    return player_contours, circles


def extend_line(line, x_start, x_end):
    a, b = np.polyfit(x=line[[0, 2]], y=line[[1, 3]], deg=1)
    y_start, y_end = [int(round(a * x + b)) for x in [x_start, x_end]]
    return [x_start, y_start, x_end, y_end]


def find_enzone_lines(img, binary_img, show_endzone_lines=cfg.show_endzone_lines):
    # parameters
    distance_res, angle_res = 1, np.pi / 180 / 2
    expected_len = int(img.shape[1] * 0.5)  # false positives are ok because of further filtering
    max_gap = int(0.2 * expected_len)
    endzone_height_px = cfg.endzone_height_m * cfg.resize_factor
    epsilon_px = int(0.1 * endzone_height_px)

    # algorithm
    lines = cv2.HoughLinesP(binary_img, distance_res, angle_res, expected_len, None, expected_len, max_gap)[:, 0]
    h_lines = lines[np.abs(lines[:, 1] - lines[:, 3]) < epsilon_px]
    ez_lines = []
    for expected_y in [endzone_height_px, img.shape[0] - endzone_height_px]:
        matches = h_lines[np.abs(h_lines[:, 1] - expected_y) < epsilon_px]
        best_match = matches[np.argmax(matches[:, 3] - matches[:, -1])]
        ez_lines.append(extend_line(best_match, 0, img.shape[1]))

    # visualization
    if show_endzone_lines:
        binary_img, img = [input_img.copy() for input_img in [binary_img, img]]
        for line_list, color in [(lines, cfg.cv2_red), (ez_lines, cfg.cv2_green)]:
            for l in line_list:
                cv2.line(img, (l[0], l[1]), (l[2], l[3]), color, 3)
        cv_utils.display_imgs([binary_img, img])
    return [(l[:2], l[2:]) for l in ez_lines]


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


def get_morph_circle(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def is_hatched(contour, binary, show=False):
    contour, binary = contour.copy(), binary.copy()
    lw = int(round(2 * cfg.resize_factor))
    mask = np.zeros_like(binary)
    cv2.drawContours(binary, [contour], -1, cfg.medium_intensity, lw)
    cv2.drawContours(mask, [contour], -1, cfg.max_intensity, cfg.filled)
    binary = cv_utils.crop_to_content(np.bitwise_and(binary, mask))
    if binary is None:
        return False
    binary = cv2.ximgproc.thinning(binary)
    thr = 10
    max_gap = int(round(1 * cfg.resize_factor))
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180 / 3, thr, None, None, max_gap)
    min_lines_hatched = 3
    if lines is None or len(lines) < min_lines_hatched:
        return False
    else:
        lines = lines[:, 0]
    if show:
        for l in lines:
            cv2.line(binary, (l[0], l[1]), (l[2], l[3]), cfg.medium_intensity, 3)
        cv_utils.display_img(binary)
    angles = [np.arctan2(x1 - x0, y1 - y0) / np.pi * 180 for x0, y0, x1, y1 in lines]
    std_angles = np.std(angles)
    max_std_angels = 70
    hatched = std_angles <= max_std_angels
    return hatched


def detect_handdrawings(gray_img, player_circles, field, disc_pos, show=cfg.show_areas):
    mask = field.copy()
    gray_img = cv2.medianBlur(gray_img, 9)
    annotated = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    cv_utils.display_img(gray_img, wait=False) if show else None
    pl_factor, disc_factor = 2.1, 1.8
    [cv2.circle(mask, (c[0], c[1]), int(c[2] * pl_factor), cfg.max_intensity, cfg.filled) for c in player_circles]
    cv2.circle(mask, disc_pos.astype(np.int32), int(cfg.radius_pixels_disc * disc_factor), cfg.max_intensity, cfg.filled)
    binary = cv_utils.adaptive_threshold(gray_img, 55, -3)  # todo use masked adaptive threshold
    binary[mask == 255] = 0
    cv_utils.display_img(binary, wait=False) if show else None
    kernel_size = cv_utils.round_to_odd(0.8 * cfg.resize_factor)  # approx. 80 cm
    binary_dilated = cv2.dilate(binary, get_morph_circle(kernel_size))
    cv_utils.display_img(binary_dilated, wait=False) if show else None
    contours, hierarchy = cv2.findContours(binary_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area_mm = 5
    min_area_px = min_area_mm * (cfg.resize_factor ** 2)
    areas = []
    [cv2.drawContours(annotated, [contour], -1, cfg.cv2_red, 2) for contour in contours]
    for contour, hierarchy_component in zip(contours, hierarchy[0]):
        if hierarchy_component[3] < 0:
            cv2.drawContours(annotated, [contour], -1, cfg.cv2_orange, 2)
            if cv2.contourArea(contour) > min_area_px and is_hatched(contour, binary):
                # contour = cv2.convexHull(contour)[:, 0]
                contour = contour[:, 0]
                cv2.drawContours(annotated, [contour], -1, cfg.cv2_green, 2)
                areas.append(contour)
    cv_utils.display_img(annotated, wait=False) if show else None
    return np.array(areas)


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