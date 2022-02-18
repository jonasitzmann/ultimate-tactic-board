import os
import shutil
import scan
import cfg
import ip_camera
from glob import glob


def state_from_photo(show=False):
    ip_camera.take_photo()
    state = scan.scan(cfg.demo_img)
    if show:
        drawer.show(drawer.draw_scene(state), wait=1000)
    return state


def main():
    # for img_path in [imgs_dir + 'ho-stack-1.jpg', imgs_dir + 'ho-stack-2.jpg']:
    ip_camera.take_photo()
    player_positions = scan.scan(cfg.demo_img)
    surface = drawer.draw_scene(player_positions)
    drawer.show(surface, wait=0)


def animate():
    img_1, img_2 = cfg.input_imgs_dir + 'ho-stack-1.jpg', cfg.input_imgs_dir + 'ho-stack-2.jpg'
    state_1 = scan.scan(img_1)
    state_2 = scan.scan(img_2)
    drawer.animate_scene([state_1, state_2, state_1], fps=60, seconds_per_step=2)


def animation_from_image_folder(folder_path, name=None):
    name = name if name is not None else os.path.basename(folder_path)
    imgs = sorted(glob(f'{folder_path}/*.jpg'))
    states = [scan.scan(img) for img in imgs]
    [drawer.show(drawer.draw_scene(state), wait=0) for state in states]
    drawer.animate_scene(states, name=name)


def create_animation(name='current_animation'):
    img_dir = cfg.input_imgs_dir + name
    shutil.rmtree(img_dir, ignore_errors=True)
    os.makedirs(img_dir)
    i = 0
    states = []
    while input('take next photo (y/n)\n') == 'y':
        filename = f'{img_dir}/{i}.jpg'
        ip_camera.take_photo(filename)
        state = scan.scan(filename)
        drawer.show(drawer.draw_scene(state))
        if input('ok? (y/n)\n') == 'y':
            i += 1
            states.append(state)
        else:
            os.remove(filename)
    drawer.animate_scene(states, name=name)


def export_state():
    state = scan.scan(f'{cfg.input_imgs_dir}/ho-stack-2.jpg')
    state.save('temp/s_2.yaml')


if __name__ == '__main__':
    # main()
    # name = 'current_animation_2'
    # animation_from_image_folder(cfg.input_imgs_dir + name, name)
    # create_animation()
    # export_state()
    player_positions = scan.scan(f'{cfg.input_imgs_dir}/ho-stack-1.jpg')
    surface = drawer.draw_scene(player_positions)
    drawer.show(surface, wait=0)
