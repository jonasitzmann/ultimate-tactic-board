import torch
import cv2
import os
import numpy as np
from numpy import floor, ceil
from sklearn.cluster import KMeans
from digit_classification import classify_image

from multiprocessing import Process

os.system('unset SESSION_MANAGER')


show_count=0
resize_factor = 12

def show(img, scale=8, window_name=None, wait=True, pos=None):
    global show_count
    scale /= resize_factor
    pos = show_count if pos is None else pos
    h, w = (np.array(img.shape[:2]) * scale).astype(np.int32)
    if window_name is None:
        window_name = f'{show_count}'
    res_x = 1920
    num_windows_horizontal = 5
    cv2.imshow(window_name, cv2.resize(img, (w, h)))
    window_x = (res_x + res_x // num_windows_horizontal) * (pos % num_windows_horizontal)
    cv2.moveWindow(window_name, window_x, 0)
    show_count += 1
    if wait:
        cv2.waitKey(0)

def main():
    img_path = 'images/board_3_small.jpg'
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    kernel_size = 9
    img_blurred = cv2.medianBlur(img, kernel_size)
    img_gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0)
    kernel = np.ones((5, 5))
    for i in range(3):
        edges = cv2.erode(edges, kernel)
        edges = cv2.dilate(edges, kernel)
    show(edges, scale=10)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hull = []
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            hull_candidate = cv2.convexHull(c)
            hull_candidate = cv2.approxPolyDP(hull_candidate, 150, True)
            if len(hull_candidate) == 4:
                hull = hull_candidate
                max_area = area

    annotated = img.copy()
    cv2.drawContours(annotated, [hull], -1, (255, 255, 0), 3)
    cv2.drawContours(annotated, hull, -1, (255, 255, 0), 10)
    show(annotated, scale=5)
    corners = hull[:, 0]
    center = np.mean(corners, dtype=np.int16, axis=0)
    angles = [np.arctan2(*(corner - center)) for corner in corners]
    sorted_corners = np.array([corner for angle, corner in sorted(zip(angles, corners))])  # clockwise from tl
    field_height, field_width = 100, 37
    new_h, new_w = field_height * resize_factor, field_width * resize_factor
    source_points = np.array(sorted_corners, dtype=np.float32)
    target_points = np.array([(new_w, new_h), (new_w, 0), (0, 0), (0, new_h)], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source_points, target_points[[1, 2, 3, 0]])
    img_transformed = cv2.warpPerspective(img, transform, (new_w, new_h))
    # show(img_transformed, wait=True)
    gray = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    gray = (gray.astype(np.float32) * 255 / gray.max()).astype(np.uint8)
    thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -15)
    # kernel_size = 5
    # saturation = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2HSV)[:, :, 1]
    # saturation_blurred = cv2.medianBlur(saturation, kernel_size)
    # ksize_erode = int(resize_factor / 2 * 0.8) * 2 + 1  # should be odd
    # eroded = cv2.erode(saturation_blurred, np.ones((ksize_erode, ksize_erode)))
    # saturation_foreground = np.clip(saturation_blurred.astype(np.float32) - eroded.astype(np.float32), 0, 255).astype(np.uint8)
    # saturation_foreground = (saturation_foreground.astype(np.float32) * 255 / saturation_foreground.max()).astype(np.uint8)
    # # show(saturation_blurred, wait=False)
    # # flooded = saturation_blurred.copy()
    # # flood_thr_1, flood_thr_2 = 200, 4
    # # margin = 1
    # # flood_color = 0
    # # for x in np.linspace(0, new_w, 3, endpoint=False):
    # #     cv2.floodFill(flooded, None, (int(x), margin), flood_color, [flood_thr_1]*1, [flood_thr_2]*1)
    # #     cv2.floodFill(flooded, None, (int(x), new_h - margin), flood_color, [flood_thr_1] * 1, [flood_thr_2] * 1)
    # # for y in np.linspace(0, new_h, 3, endpoint=False):
    # #     cv2.floodFill(flooded, None, (margin, int(y)), flood_color, [flood_thr_1]*1, [flood_thr_2]*1)
    # #     cv2.floodFill(flooded, None, (new_w - margin, int(y)), flood_color, [flood_thr_1] * 1, [flood_thr_2] * 1)
    # # flooded[flooded > 0] = 255
    # ksize = int(resize_factor / 2 * 12) * 2 + 1
    # c = - ksize // 8
    # thresholded = cv2.adaptiveThreshold(saturation_foreground, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ksize, c)
    #
    # # find circles
    radius_cm = 1  # cm
    radius_pixels = radius_cm * new_h // field_height
    min_radius, max_radius = (np.array([0.7, 0.95]) * radius_pixels).astype(np.int32)
    circles = cv2.HoughCircles(thresholded, cv2.HOUGH_GRADIENT, 1.6, new_h / 100, param1=200, param2=20, minRadius=min_radius, maxRadius=max_radius)

    circles = np.uint16(np.around(circles))
    mask = np.zeros_like(thresholded)
    for i in circles[0, :]:
        cv2.circle(mask, (i[0], i[1]), int(radius_pixels), 255, -1)

    show(thresholded, wait=False)
    show(mask)
    # img_hsv = cv2.cvtColor(flooded, cv2.COLOR_BGR2HSV)
    annotated = img_transformed.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        try:
            number = get_number(img_transformed, c, resize_factor)
        except ValueError:
            print('error')
            continue
        one_contour = np.zeros_like(mask)
        cv2.drawContours(one_contour, contours, i, 255, -1)
        median_color = np.median(img_transformed[one_contour == 255], axis=0)
        light_color = (median_color + np.array([255]*3)) // 2
        cv2.drawContours(annotated, contours, i, light_color, -1)
        cv2.drawContours(annotated, contours, i, light_color, 8)
        center = (np.mean(c, axis=0).astype(np.int32)[0] + np.array([-10, 10])).astype(np.int32)
        font_size = 0.15 * resize_factor
        cv2.putText(annotated, str(number), center, 1, font_size, (0, 0, 0), 2)
    # show_all = False
    show_all = True
    if show_all:
        # show(img_transformed, wait=False)
        # show(saturation, wait=False)
        # show(saturation_foreground, wait=False)
        # show(thresholded, wait=False)
        # show(mask, wait=False)
        show(annotated)


    cv2.waitKey(0)
    offense_positions = []
    for cnt in contours:
        # compute the center of the contour
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        offense_positions += [(cX // resize_factor, field_height - (cY // resize_factor))]
        cv2.circle(img_transformed, (cX, cY), 7, (255, 0, 255), -1)
    mask = cv2.inRange(img_transformed, (0, 0, 0), (90, 90, 90))
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    disc_contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    disc_contour_areas = [cv2.contourArea(c) for c in disc_contours]
    max_idx = np.argmax(disc_contour_areas)
    disc_contour = disc_contours[max_idx]
    # cv2.drawContours(img_transformed, disc_contours, max_idx, (100, 200, 0), 1)
    M_disc = cv2.moments(disc_contour)
    disc_x = int(M_disc["m10"] / M_disc["m00"])
    disc_y = int(M_disc["m01"] / M_disc["m00"])
    disc_position = (disc_x // resize_factor, field_height - disc_y // resize_factor)

    # cv2.imshow('', img_transformed)
    # cv2.waitKey(0)
    # print(disc_contours)
    # print(offense_positions)
    # return offense_positions, defense_positions, disc_position


def get_number(image, contour, scale):
    image = image.copy()
    mask = np.zeros_like(image[:, :, 0])
    cv2.drawContours(mask, [contour], 0, 255, -1)
    # show(crop_to_content(mask, image), 100, 'crop_original', False, 0)
    median_color = np.median(image[mask == 255], axis=0).astype(np.uint8)
    image[mask == 0] = median_color
    img_blurred = image.copy()
    n_blurrs = 5
    for i in range(n_blurrs):
        img_blurred = cv2.GaussianBlur(img_blurred, (3, 3), 5)
        img_blurred[mask == 255] = image[mask == 255]
    img_sharp = np.clip(img_blurred.astype(np.int32) - cv2.GaussianBlur(img_blurred, (7, 7), 3).astype(np.int32), 0, 255).astype(np.uint8)
    max_color = img_sharp.reshape(-1, 3).max()
    img_sharp = (img_sharp.astype(np.float32) * 255 / max_color).astype(np.uint8)
    crop = crop_to_content(mask, img_sharp)
    mask = crop_to_content(mask, mask)
    crop[mask == 0] *= 0
    show(crop, 100, 'digit_exposed', wait=False, pos=2)
    h, w = crop.shape[:2]
    crop_flat = crop.reshape(-1, 3)
    mask_flat = mask.reshape(-1, 1)
    kmeans = KMeans(2)
    kmeans.fit(crop_flat[mask_flat.reshape(-1) == 255, :])
    # kmeans.fit(crop_flat)
    kmeans.cluster_centers_ = np.array(sorted(kmeans.cluster_centers_, key=sum, reverse=False))
    crop_flat_binary = np.array(kmeans.predict(crop_flat))
    crop_binary = crop_flat_binary.reshape(h, w).astype(np.uint8) * 255
    # crop_binary = cv2.erode(crop_binary, np.ones((2, 2)))
    crop_binary = cv2.bitwise_and(crop_binary, mask)
    crop_binary = crop_to_content(crop_binary, crop_binary)
    longer_side = max(crop_binary.shape[:2])
    crop_padded = np.pad(crop_binary, longer_side)
    target_size = 28
    preds = []
    entropies = []
    # show(crop)
    # show(crop_binary)
    for angle in np.linspace(0, 360, 1, endpoint=False):
        crop_rotated = rotate_image(crop_padded, angle)
        crop_rotated = crop_to_content(crop_rotated, crop_rotated)
        longer_side = max(crop_rotated.shape[:2])
        resize_factor = (target_size - 6) / longer_side
        new_size = (np.array(crop_rotated.shape[:2]) * resize_factor).astype(np.int32)
        new_size = np.array([new_size[1], new_size[0]])
        # cv2.imshow('crop_binary', crop_binary)
        crop_rotated = cv2.resize(crop_rotated, new_size)
        pad_lr, pad_tb = (target_size - new_size) / 2
        pad_sizes = np.array([ceil(pad_tb), floor(pad_tb), ceil(pad_lr), floor(pad_lr)], dtype=np.int32)
        crop_rotated = cv2.copyMakeBorder(crop_rotated, *pad_sizes, cv2.BORDER_CONSTANT)
        # show(crop_rotated, 80, 'like mnist', True, pos=3)
        classification = classify_image(crop_rotated)
        preds.append(classification[0])
        entropies.append(classification[1])
        # cv2.imshow('crop_final', crop_rotated)
        # cv2.waitKey(0)
    return preds[0]


def crop_to_content(mask, image_to_crop):
    (y, x) = np.where(mask)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    crop = image_to_crop[topy: bottomy+1, topx:bottomx+1]
    return crop


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

if __name__ == '__main__':
    main()