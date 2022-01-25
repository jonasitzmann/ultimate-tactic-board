import cairo
import cv2
import numpy as np
from typing import Optional
import export_scene
import os
import shutil
import state
import cfg


def main():
    init_context()
    draw_field()
    draw_player(10, 0)
    show(surface)


def draw_scene(state: state.State):
    init_context()
    draw_field()
    [draw_player(player, (0.6, 0, 0)) for player in state.players_team_1]
    [draw_player(player, (0, 0, 0.6)) for player in state.players_team_2]
    return surface


def animate_scene(state_1, state_2, num_steps, name='animation'):
    output_path = f'{cfg.media_out_dir}{name}'
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)
    for i, frac in enumerate(np.linspace(0, 1, num_steps)):
        state = state_1 * (1 - frac) + state_2 * frac
        surface = draw_scene(state)
        path = f'{output_path}/{i:04d}.png'
        surface.write_to_png(path)
    export_scene.save_video(output_path, fps=10)
    shutil.rmtree(output_path)


# globals
width, height, endzone_height = 37, 100, 18
border = 3
scale = 8
ctx: Optional[cairo.Context] = None
surface: Optional[cairo.ImageSurface] = None


def draw_field():
    draw_background()
    ctx.move_to(*m2p([border, border]))
    select_line_brush()
    path = [[width, 0], [0, height], [-width, 0], [0, -height]]
    for line in path:
        rel_line(*line)
    for y in [endzone_height, height - endzone_height]:
        move_to(0, y)
        rel_line(width, 0)
    ctx.stroke()


def draw_background():
    ctx.set_source_rgb(0.3, 0.5, 0.3)
    ctx.move_to(*m2p([0, 0]))
    w = width + 2 * border
    h = height + 2 * border
    path = [[w, 0], [0, h], [-w, 0], [0, -h]]
    for line in path:
        rel_line(*line)
    ctx.fill()

@np.vectorize
def m2p(x, rounded=True):
    x = scale * x
    if rounded:
        x = int(x)
    return x


def move_to(x, y):
    ctx.move_to(*m2p([border + x, border + y]))


def rel_line(x, y):
    ctx.rel_line_to(*m2p([x, y]))


def show(surface, filename='temp', wait=10):
    os.makedirs(cfg.media_out_dir, exist_ok=True)
    path = f'{cfg.media_out_dir}/{filename}.png'
    surface.write_to_png(path)
    cv2.imshow(filename, cv2.imread(path))
    window_x = cfg.resolution_x - 50 - m2p(width + 2*border)
    cv2.moveWindow(filename, window_x, 0)
    cv2.waitKey(wait)


def select_line_brush():
    ctx.set_source_rgb(0.7, 0.7, 0.7)  # Solid color
    ctx.set_line_width(m2p(0.3))


def init_context():
    global ctx, surface
    size_meters = np.array([width + 2 * border, height + 2 * border])
    size_pixels = m2p(size_meters / 2, rounded=True) * 2
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, *size_pixels)
    ctx = cairo.Context(surface)


def draw_player(player: state.Player, color):
    ctx.set_line_width(0)
    ctx.set_source_rgb(*color)
    radius = m2p(1, rounded=False)
    move_to(*player.pos)
    ctx.arc(*ctx.get_current_point(), radius, 0, 2 * np.pi)
    ctx.fill()
    ctx.set_source_rgb(0.8, 0.8, 0.8)
    ctx.select_font_face("Serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(m2p(2))
    pos = [m2p(player.pos[0] + border), m2p(player.pos[1] + border)]
    text(player.label, pos, player.orientation)


def text(string, pos, orientation):
    ctx.save()
    fascent, fdescent, fheight, fxadvance, fyadvance = ctx.font_extents()
    x_off, y_off, tw, th = ctx.text_extents(string)[:4]
    nx = -tw/2
    ny = fheight/2
    ctx.translate(pos[0], pos[1])
    ctx.rotate(orientation / -180 * np.pi)
    ctx.translate(nx, ny)
    ctx.move_to(0, -4)
    ctx.show_text(string)
    ctx.restore()


if __name__ == "__main__":
    main()
