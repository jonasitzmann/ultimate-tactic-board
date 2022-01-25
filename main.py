import scan
import drawer
images_dir = 'input_images/'


def main():
    for image_path in [images_dir + 'ho-stack-1.jpg', images_dir + 'ho-stack-2.jpg']:
    # image_path = images_dir + 'disc.jpg'
        show_digits = False
        show_circles = True
        record_examples = False
        player_positions = scan.scan(image_path, show_digits, show_circles, record_examples)
        surface = drawer.draw_scene(player_positions)
        drawer.show(surface, wait=0)


def animate():
    img_1, img_2 = images_dir + 'ho-stack-1.jpg', images_dir + 'ho-stack-2.jpg'
    show_circles = False
    show_digits = False
    state_1 = scan.scan(img_1, show_digits, show_circles)
    state_2 = scan.scan(img_2, show_digits, show_circles)
    drawer.animate_scene(state_1, state_2, 30)


if __name__ == '__main__':
    main()
    # animate()
