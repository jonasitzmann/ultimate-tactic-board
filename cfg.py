# coding=utf-8
# directories
input_imgs_dir = 'input_imgs/'
media_out_dir = 'media_out/'
data_dir = 'data/'
reference_digits_dir = data_dir + 'reference_digits/'

# dimensions
resize_factor = 20
field_height_m, field_width_m = 100, 37  # dimensions of an actual field, not the tactics board
resolution_x, resolution_y = 1920, 1080
max_num_windows = 5
digit_target_size = 28

#
min_intensity, max_intensity, medium_intensity = 0, 255, 100
cv2_black = tuple([min_intensity] * 3)
ksize_sharpening = 0.8
ksize_blur_thresholded = 3

# field detection
ksize_initial_blur = 15
field_detection_poly_epsilon = 150
offset_binarize_global = -3

# player detection
ksize_blur_crop = 11


radius_players_cm = 1
player_radius_lb, player_radius_ub = 0.75, 0.9

# digits
font_size = 0.15 * resize_factor
