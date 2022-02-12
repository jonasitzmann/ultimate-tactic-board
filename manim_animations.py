from contextlib import contextmanager
import ultimate_scene

import vlc
from cv_utils import round_to_odd
import cfg
from manim import *
from glob import glob
import inspect
import os
import subprocess
from state import State
from functools import partial
landscape_scale = (config.frame_x_radius - DEFAULT_MOBJECT_TO_EDGE_BUFFER) / 50  # landscape
portrait_scale = (config.frame_y_radius - DEFAULT_MOBJECT_TO_EDGE_BUFFER) / 50  # portrait
portrait_scale_no_buffer = config.frame_y_radius / 50

myred = '#c22a2a'
myblue = '#44496f'

class MState(VGroup):
    def __init__(self, state, cs=None):
        self.cs = cs
        self.offense, self.defense, self.disc = VDict(), VDict(), Mobject()
        self.state = state
        if cs is not None:
            self.update_cs()
        super().__init__(self.offense, self.defense)

    def update_cs(self, cs=None):
        if cs is not None:
            self.cs = cs
        for p in self.state.players_team_1:
            self.offense[p.label] = MPlayer(role='o', angle=p.angle * DEGREES - PI, position=p.manimpos,
                                            frame_of_ref=self.cs)
        for p in self.state.players_team_2:
            self.defense[p.label] = MPlayer(role='d', angle=p.angle * DEGREES - PI, position=p.manimpos,
                                            frame_of_ref=self.cs)
        if len(self.state.disc.shape) == 1:
            self.disc = Dot(
                stroke_width=0,
                fill_color=WHITE,
                fill_opacity=1,
                radius=0.5 * self.cs.current_scale(),
            ).shift(self.cs.c2p(*self.state.disc))
            self.disc.z_index = 3
            self.add(self.disc)

    def get_animations(self, new_state):
        movements = []
        for current, target in [(self.offense, new_state.offense), (self.defense, new_state.defense)]:
            for k, v in current.submob_dict.items():
                t = target.submob_dict.get(k, None)
                if t is not None:
                    movements.append(MovePlayer(v, t, self.cs))
        return movements


    @contextmanager
    def marker_shadow(self, scene):
        shadow = always_redraw(self.get_marking_shadow)
        scene.add(shadow)
        scene.play(FadeIn(shadow), run_time=0.8)
        yield
        scene.play(FadeOut(shadow), run_time=0.8)
        scene.remove(shadow)

    def get_marking_shadow(self):
        marking_reach = 2 * self.cs.current_scale()
        offender = sorted(self.offense.submob_dict.values(), key=lambda x: np.linalg.norm(x.position[:2] - self.state.disc))[0]
        marker = sorted(self.defense.submob_dict.values(), key=lambda x: np.linalg.norm(x.position - offender.position))[0]
        o_pos, d_pos = offender.get_center(), marker.get_center()
        marking_shadow_size = 30 * self.cs.current_scale()
        marking_line = Line(d_pos + 0.5 * marking_reach * LEFT, d_pos + 0.5 * marking_reach * RIGHT)
        marking_line.rotate(marker.angle + self.cs.get_angle())
        pts = [marking_line.get_start(), marking_line.get_end()]
        for pt in pts[::-1]:
            direction = Line(o_pos, pt).get_unit_vector()
            pts.append(o_pos + direction * marking_shadow_size)
        area = Polygon(*pts)
        area = Intersection(area, self.cs.rect, stroke_width=0, fill_color=marker.get_color(), fill_opacity=0.3)
        area.z_index = 1
        return area



class Field(VGroup):
    def __init__(self, scene=None, state=None, height=10, scale_for_landscape=True, **kwargs):
        VGroup.__init__(self, **kwargs)
        scale = height / 100
        self.cs = FrameOfReference(100, 37, scale)
        self.add(self.cs)
        self.s = state or MState(cs=self.cs)
        h, w = np.array([100, 37]) * scale
        ez_height = 18 * scale
        self.add(Rectangle(height=h, width=w, fill_color='#728669', fill_opacity=0.7))
        for direction, color in zip([UP, DOWN], [myred, myblue]):
            self.add(Line(ORIGIN, w * RIGHT, color=color).shift(w / 2 * LEFT + (h / 2 - ez_height) * direction))
            self.add(Cross(stroke_color=WHITE, scale_factor=0.05, stroke_width=3).shift((h/2 - 2*ez_height) * direction))
        self.rect = Rectangle(height=h, width=w)
        self.add(self.rect)
        self.load(state)
        if scale_for_landscape:
            self.scale((config.frame_x_radius - DEFAULT_MOBJECT_TO_EDGE_BUFFER) / 5).rotate(90 * DEGREES)
        if scene is not None:
            scene.add(self)

    def landscape_to_portrait(self, scene):
        scene.play(self.animate.scale(ls2pt).rotate(90 * DEGREES).move_to(ORIGIN).to_edge(RIGHT))

    def load(self, state: MState):
        self.remove(self.s)
        self.s = state
        self.s.update_cs(self.cs)
        self.add(self.s)

    def transition(self, scene, s2: MState, run_time=4):
        s2.update_cs(self.cs)
        scene.play(*self.s.get_animations(s2), run_time=run_time)


class FrameOfReference(Axes):
    def __init__(self, h, w, scale, *args, **kwargs):
        super().__init__(x_range=[0, w, w+1], y_range=[0, h, h+1], x_length=w*scale, y_length=h*scale, tips=False)
        self.c2p = self.coords_to_point

    def current_scale(self):
        return np.linalg.norm((self.c2p(*UP) - self.c2p(*ORIGIN)))

    def get_angle(self):
        pt = (self.c2p(*UP) - self.c2p(*ORIGIN))[:2]
        orientation = - np.arctan2(*pt)
        return orientation

    @property
    def rect(self):
        x1, x2 = self.x_range[:2]
        y1, y2 = self.y_range[:2]
        return Polygon(self.c2p(x1, y1), self.c2p(x2, y1), self.c2p(x2, y2), self.c2p(x1, y2))


class MPlayer(VGroup):
    def __init__(self, angle, position, frame_of_ref: FrameOfReference, color=None, role='d'):
        size = np.array([1.2, 1.2] if role == 'o' else [2, 0.3]) * frame_of_ref.current_scale()
        self.role = role
        if color is None:
            color = myblue if self.is_o else myred
        super().__init__(color=color)
        self.ellipse=Ellipse(*size, color=color, fill_color=color, fill_opacity=1, z_index=2)
        self.angle = angle
        self.position = position
        self.frame_of_ref = frame_of_ref
        self.nose = Triangle(color=color, z_index=2).scale_to_fit_height(0.3*frame_of_ref.current_scale()).move_to(size[1]/2*UP, aligned_edge=DOWN)
        self.add(self.ellipse, self.nose)
        self.set_pose(position, angle)
    @property
    def is_o(self):
        return self.role == 'o'

    def set_pose(self, position, angle):
        self.rotate(angle + self.frame_of_ref.get_angle())
        self.move_to(self.frame_of_ref.c2p(*position))

    def get_highlight(self):
        hightlight_opacity = 0.3
        c = Circle(
            radius=3 * self.frame_of_ref.current_scale(),
            fill_color=self.get_color(),
            stroke_width=0,
            fill_opacity=hightlight_opacity,
        ).shift(self.get_center() + IN)
        return c






    @contextmanager
    def highlight(self, scene):
        c = self.get_highlight()
        scene.play(FadeIn(c), run_time=0.8)
        self.add(c)
        yield
        scene.play(FadeOut(c), run_time=0.8)
        self.remove(c)


def get_minimum_rotation(radians):
    left = radians % TAU
    right = left - TAU
    return left if abs(left) < abs(right) else right


ls2pt = portrait_scale / landscape_scale



class MovePlayer(Animation):
    def __init__(self, mobject, target, cs: FrameOfReference, real_time=4, *args, **kwargs):
        super().__init__(mobject, *args, **kwargs)
        self.end_pose = target
        self.radians = get_minimum_rotation(target.angle - mobject.angle)
        self.cs = cs
        avg_speed_kmh = np.linalg.norm(target.position - mobject.position) / real_time * 3.6
        max_speed_floating = 50
        if avg_speed_kmh > max_speed_floating:
            self.interpolate_mobject = self.beam_interpolation  # todo: how to interpolate fast players?
        else:
            self.interpolate_mobject = self.float_interpolation

    def beam_interpolation(self, alpha):
        self.mobject.become(self.starting_mobject if self.rate_func(alpha) < 0.5 else self.end_pose)

    def float_interpolation(self, alpha: float) -> None:
        self.mobject.become(self.starting_mobject)
        k = self.rate_func(alpha)
        self.mobject.angle = self.starting_mobject.angle + k * self.radians
        self.mobject.position = self.starting_mobject.position + k * (self.end_pose.position - self.starting_mobject.position)
        self.mobject.rotate(k * self.radians)
        pos = self.cs.c2p(*((1 - k) * self.mobject.position) + k * self.end_pose.position)
        self.mobject.move_to(pos)

def create_movie(cls, debug=False, opengl=False, hq=False, output_file=None, play=False):
    if debug:
        obj = cls()
        obj.render(True)
    else:
        cls_name, cls_path = cls.__name__, inspect.getfile(cls)
        opengl_part = '--renderer=opengl' if opengl else ''
        output_file = cfg.default_animation_file if output_file is None else output_file
        ouput_part = '-o ../../../../' + output_file
        command = f'{opengl_part} {ouput_part} -pq{"h" if hq else "l"} {cls_path} {cls_name}'
        manim_path = '/home/jonas/.conda/envs/tactics_board/bin/manim'
        # subprocess.call((manim_path, command))
        # if play:
        #     subprocess.call(('vlc', output_file))
        os.system(f'{manim_path} {command}')


class StateImg(Scene):
    @staticmethod
    def get(state=None):
        if state is None:
            state = State.load('temp/s_copy.yaml')
        scene = StateImg(state)
        scene.render()
        img_path = scene.renderer.file_writer.image_file_path
        print(img_path)
        return str(img_path)


    def __init__(self, state, *args, **kwargs):
        self.state = state
        config.pixel_width = 1500
        fh = config.frame_width
        config.frame_height = fh * 37 / 100
        config.pixel_height = round_to_odd(config.pixel_width * 37 / 100) + 1
        config.frame_height = config.frame_width * 37 / 100
        super().__init__(*args, **kwargs)

    def construct(self):
        s1 = MState(self.state)
        self.add(Field(state=s1, height=10, scale_for_landscape=False).scale(config.frame_width / 10).rotate(90 * DEGREES))


def get_states_from_dir(dir_path):
    sort_func = lambda filepath: f'{int(filepath.split("/")[-1].split(".")[0]):02d}'
    states = [MState(State.load(f)) for f in
                   sorted([f for f in glob(f'{dir_path}/*.yaml')], key=sort_func)]
    return states


class DirAnimation(Scene):
    def __init__(self, *args, **kwargs):
        self.states = get_states_from_dir(cfg.current_play_dir)
        config.pixel_height = 1080
        config.pixel_width = 1920
        config.frame_height = 8
        config.frame_width = 8 * 1920 / 1080
        super().__init__(*args, **kwargs)

    def construct(self):
        field = Field(state=self.states[0], height=10).scale(config.frame_width / 10).rotate(90 * DEGREES)
        self.add(field)
        for s1, s2 in zip(self.states, self.states[1:]):
            field.load(s1)
            animations = field.transition(s1, s2)
            self.play(*animations, run_time=3)


def animation_from_state_dir(output_file=None, play=False):
    create_movie(DirAnimation, output_file=output_file, play=play)

if __name__ == '__main__':
    # animation_from_state_dir(cfg.default_animation_file)
    # StateImg.get()
    debug = True
    hq = False
    opengl = False
    create_movie(ultimate_scene.UltimateScene, debug=debug, hq=hq, opengl=opengl, play=True)
    # create_movie(LocalFrameOfReference)

