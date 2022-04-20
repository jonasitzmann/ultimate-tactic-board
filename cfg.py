import cv2
# todo: a hierarchical config file (e.g. yaml) would be better suited

# coding=utf-8
# directories
input_imgs_dir = 'input_imgs/'
media_out_dir = 'media_out/'
data_dir = 'data/'
reference_digits_dir = data_dir + 'reference_digits/'
temp_dir = 'temp/'
plays_dir = 'plays/'
current_play_dir = plays_dir + 'current/'
default_animation_file = f'{temp_dir}animation.mp4'
template_dir = 'templates/'
template_play_file = template_dir + 'play_template.py'
logo_path = data_dir + 'logo.png'


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
cv2_red = (min_intensity, min_intensity, max_intensity)
cv2_yellow = (min_intensity, max_intensity, max_intensity)
cv2_orange = (min_intensity, 150, max_intensity)
cv2_green = (min_intensity, max_intensity, min_intensity)
ksize_sharpening = 0.8
ksize_blur_thresholded = 3

# field detection
ksize_initial_blur = 11
field_detection_poly_epsilon = 150
offset_binarize_global = -3
ksize_thresh_field = 7
offset_thresh_field = -1

# player detection
ksize_blur_crop = 11
filled = -1

# frame
ksize_thresh_frame = 45
offset_thresh_frame = -2

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



# open camera remote server
ip = '192.168.178.22'
http_port = 8080
udp_port = 8000
camera_save_path = input_imgs_dir + 'current.jpg'
img_prefix = 'IMG_'
autofocus_time_seconds = 1  # delay to wait for autofocus


# options
show_digits = False
show_circles = False
labeling_mode = False
show_edges = False
show_endzone_lines = False
show_median_color = False
show_field = False
show_areas = False
show_arrows = False
show_input = False
show_transformed = False
# demo_img = input_imgs_dir + '/not_supported/' + 'bad_perspective.jpg'
# demo_img = input_imgs_dir + '/not_supported/' + 'better_perspective.jpg'
# demo_img = input_imgs_dir + '/not_supported/' + 'better_perspective_rotated.jpg'
# demo_img = input_imgs_dir + '/not_supported/' + 'all_together.jpg'
# demo_img = input_imgs_dir + '/not_supported/' + 'arrows_and_areas.jpg'
# demo_img = input_imgs_dir + '/not_supported/' + 'new_arrows.jpg'
demo_img = input_imgs_dir + 'current.jpg'
demo_state_path = temp_dir + 's.yaml'

#manim
o_width = 1.2
o_height = 1.2
d_width = 2
d_height = 0.5
player_scale = 1

# tactics
hex_dist_m = 11