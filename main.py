import scan
import drawer
imgs_dir = 'input_imgs/'


def main():
    # for img_path in [imgs_dir + 'ho-stack-1.jpg', imgs_dir + 'ho-stack-2.jpg']:
    img_path = imgs_dir + 'disc.jpg'
    show_digits = True
    show_circles = True
    record_examples = False
    player_positions = scan.scan(img_path, show_digits, show_circles, record_examples)
    surface = drawer.draw_scene(player_positions)
    drawer.show(surface, wait=0)


def animate():
    img_1, img_2 = imgs_dir + 'ho-stack-1.jpg', imgs_dir + 'ho-stack-2.jpg'
    show_circles = False
    show_digits = False
    state_1 = scan.scan(img_1, show_digits, show_circles)
    state_2 = scan.scan(img_2, show_digits, show_circles)
    drawer.animate_scene(state_1, state_2, 30)


if __name__ == '__main__':
    # main()
    animate()
