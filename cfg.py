import cv2
# todo: a hierarchical config file (e.g. yaml) would be better suited

# coding=utf-8
# directories
input_imgs_dir = 'input_imgs/'
media_out_dir = 'media_out/'
data_dir = 'data/'
reference_digits_dir = data_dir + 'reference_digits/'

# dimensions
resize_factor = 20
field_height_m, field_width_m, endzone_height_m = 100, 37, 18  # dimensions of an actual field, not the tactics board
resolution_x, resolution_y = 1920, 1080
max_num_windows = 5
digit_target_size = 28
tactic_board_height_cm, tactic_board_width_cm = 100, 37
radius_players_cm = 1
radius_disc_cm = 0.75
player_radius_lb, player_radius_ub = 0.75, 0.9
disc_radius_lb, disc_radius_ub = 0.8, 1.2
radius_pixels = radius_players_cm * resize_factor * field_height_m // tactic_board_height_cm
radius_pixels_disc = int(radius_disc_cm * resize_factor * field_height_m // tactic_board_height_cm)

#
min_intensity, max_intensity, medium_intensity = 0, 255, 100
cv2_black = tuple([min_intensity] * 3)
ksize_sharpening = 0.8
ksize_blur_thresholded = 3

# field detection
ksize_initial_blur = 11
field_detection_poly_epsilon = 150
offset_binarize_global = -3
ksize_thresh_field = 15
offset_thresh_field = 1

# player detection
ksize_blur_crop = 11
filled = -1

# frame
ksize_thresh_frame = 45
offset_thresh_frame = -10

h_circles_args = dict(method=cv2.HOUGH_GRADIENT, dp=1.8, param1=150, param2=25)
h_circles_args_disc = dict(method=cv2.HOUGH_GRADIENT, dp=1.8, param1=150, param2=20)

# digits
font_size = 0.15 * resize_factor
ksize_thresh_digits = 17
offset_thresh_digits = -6
min_contour_area_digit = 20



# drawing
border_size_m = 3
draw_scale = 8.7
disc_size_m = 0.6

# options
show_digits = False
show_circles = True
labeling_mode = False
show_edges = False
# demo_img = input_imgs_dir + '/not_supported/' + 'bad_perspective.jpg'
demo_img = input_imgs_dir + '/not_supported/' + 'better_perspective.jpg'
# demo_img = input_imgs_dir + '/not_supported/' + 'all_together.jpg'
