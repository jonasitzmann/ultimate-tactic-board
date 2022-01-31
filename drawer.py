import cairo
import cv2
import numpy as np
from typing import Optional
import export_scene
import os
import shutil
import state
import cfg
import cv_utils


def main():
    init_context()
    draw_field()
    draw_player(10, 0)
    show(surface)


def draw_scene(state: state.State):
    init_context()
    draw_background()
    draw_field()
    if state.areas is not None:
        [draw_area(area, (0.7, 0.2, 0.2, 0.3)) for area in state.areas]
    [draw_player(player, (0.6, 0, 0)) for player in state.players_team_1]
    [draw_player(player, (0, 0, 0.6)) for player in state.players_team_2]
    if state.disc is not None:
        draw_disc(state.disc)
    return surface


def draw_area(pts, color):
    pts = m2p(pts, add_border=True)
    ctx.set_line_width(0)
    ctx.set_source_rgba(*color)
    ctx.move_to(*pts[0])
    for x, y in pts:
        ctx.line_to(x, y)
    ctx.close_path()
    ctx.fill()


def draw_disc(disc_pos):
    ctx.set_line_width(0)
    ctx.set_source_rgb(1, 1, 1)
    radius = m2p(cfg.disc_size_m, rounded=False)
    move_to(*disc_pos)
    pos = ctx.get_current_point()
    ctx.arc(*pos, radius, 0, 2 * np.pi)
    ctx.fill()


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


ctx: Optional[cairo.Context] = None  # singleton
surface: Optional[cairo.ImageSurface] = None  # singleton


def draw_field():
    ctx.move_to(*m2p([cfg.border_size_m, cfg.border_size_m]))
    select_line_brush()
    path = [[cfg.field_width_m, 0], [0, cfg.field_height_m], [-cfg.field_width_m, 0], [0, -cfg.field_height_m]]
    for line in path:
        rel_line(*line)
    for y in [cfg.endzone_height_m, cfg.field_height_m - cfg.endzone_height_m]:
        move_to(0, y)
        rel_line(cfg.field_width_m, 0)
    ctx.stroke()


def draw_background():
    ctx.set_source_rgb(0.3, 0.5, 0.3)
    ctx.move_to(*m2p([0, 0]))
    w = cfg.field_width_m + 2 * cfg.border_size_m
    h = cfg.field_height_m + 2 * cfg.border_size_m
    path = [[w, 0], [0, h], [-w, 0], [0, -h]]
    for line in path:
        rel_line(*line)
    ctx.fill()


@np.vectorize
def m2p(x, rounded=True, add_border=False):
    if add_border:
        x += cfg.border_size_m
    x = cfg.draw_scale * x
    if rounded:
        x = int(x)
    return x


def move_to(x, y):
    ctx.move_to(*m2p([cfg.border_size_m + x, cfg.border_size_m + y]))


def rel_line(x, y):
    ctx.rel_line_to(*m2p([x, y]))


def show(surface, filename='temp', wait=10, pos=4):
    os.makedirs(cfg.media_out_dir, exist_ok=True)
    path = f'{cfg.media_out_dir}/{filename}.png'
    surface.write_to_png(path)
    cv_utils.display_img(cv2.imread(path), filename, False, pos)
    cv2.waitKey(wait)


def select_line_brush():
    ctx.set_source_rgb(0.7, 0.7, 0.7)  # Solid color
    ctx.set_line_width(m2p(0.3))


def init_context():
    global ctx, surface
    size_meters = np.array([cfg.field_width_m + 2 * cfg.border_size_m, cfg.field_height_m + 2 * cfg.border_size_m])
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
    pos = [m2p(player.pos[0] + cfg.border_size_m), m2p(player.pos[1] + cfg.border_size_m)]
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
