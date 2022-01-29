import scan
import drawer
import cfg
imgs_dir = 'input_imgs/'


def main():
    # for img_path in [imgs_dir + 'ho-stack-1.jpg', imgs_dir + 'ho-stack-2.jpg']:
    player_positions = scan.scan(cfg.demo_img)
    surface = drawer.draw_scene(player_positions)
    drawer.show(surface, wait=0)


def animate():
    img_1, img_2 = imgs_dir + 'ho-stack-1.jpg', imgs_dir + 'ho-stack-2.jpg'
    state_1 = scan.scan(img_1)
    state_2 = scan.scan(img_2)
    drawer.animate_scene(state_1, state_2, 30)


if __name__ == '__main__':
    main()
    # animate()
