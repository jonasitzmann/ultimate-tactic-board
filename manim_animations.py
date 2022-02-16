from contextlib import contextmanager, ExitStack
import scenes
from functools import partial

from cv_utils import round_to_odd
import cfg
from manim import *
from glob import glob
import inspect
import os
from state import State
landscape_scale = (config.frame_x_radius - DEFAULT_MOBJECT_TO_EDGE_BUFFER) / 50  # landscape
portrait_scale = (config.frame_y_radius - DEFAULT_MOBJECT_TO_EDGE_BUFFER) / 50  # portrait
portrait_scale_no_buffer = config.frame_y_radius / 50

myred = '#c22a2a'
myblue = '#44496f'


def get_trapezoid(vanishing_pt, shorter_side_pt_1, shorter_side_pt_2, distance):
    pts = [shorter_side_pt_1, shorter_side_pt_2]
    for pt in pts[::-1]:
        direction = Line(vanishing_pt, pt).get_unit_vector()
        pts.append(vanishing_pt + direction * distance)
    return Polygon(*pts)



class MState(VGroup):
    def __init__(self, state, cs=None):
        self.cs = cs
        self.offense, self.defense, self.disc = VDict(), VDict(), None
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
            self.disc.z_index = 2
            self.add(self.disc)

    def get_animations(self, new_state, disc_delay=0.5):
        movements = []
        for current, target in [(self.offense, new_state.offense), (self.defense, new_state.defense)]:
            for k, v in current.submob_dict.items():
                t = target.submob_dict.get(k, None)
                if t is not None:
                    movements.append(MovePlayer(v, t, self.cs))
        # if self.disc is not None and new_state.disc is not None:
        #     movements.append(MoveDisc(self.disc, new_state.disc, disc_delay))
        return movements


    @contextmanager
    def marker_shadow(self):
        shadow = always_redraw(self.get_marking_shadow)
        global_scene.add(shadow)
        global_scene.play(FadeIn(shadow), run_time=0.8)
        yield
        global_scene.play(FadeOut(shadow), run_time=0.8)
        global_scene.remove(shadow)

    def get_marking_shadow(self):
        marking_reach = 2 * self.cs.current_scale()
        offender = sorted(self.offense.submob_dict.values(), key=lambda x: np.linalg.norm(x.position[:2] - self.state.disc))[0]
        marker = sorted(self.defense.submob_dict.values(), key=lambda x: np.linalg.norm(x.position - offender.position))[0]
        o_pos, d_pos = offender.get_center(), marker.get_center()
        marking_shadow_size = 30 * self.cs.current_scale()
        marking_line = Line(d_pos + 0.5 * marking_reach * LEFT, d_pos + 0.5 * marking_reach * RIGHT)
        marking_line.rotate(marker.angle + self.cs.get_angle())
        area = get_trapezoid(o_pos, marking_line.get_start(), marking_line.get_end(), marking_shadow_size)
        area = Intersection(area, self.cs.rect, stroke_width=0, fill_color=marker.get_color(), fill_opacity=0.3)
        area.z_index = 1
        return area

    def get_player(self, player_name):
        role, label = player_name[0], player_name[1:]
        plist = self.offense if role == 'o' else self.defense
        return plist[label]




global_scene = None

class Field(VGroup):
    global scene
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
            self.scale((config.frame_x_radius - DEFAULT_MOBJECT_TO_EDGE_BUFFER) / 5).rotate(-90 * DEGREES)
        global global_scene
        global_scene = scene
        if global_scene is not None:
            global_scene.add(self)

    def landscape_to_portrait(self):
        global_scene.play(self.animate.scale(ls2pt).rotate(-90 * DEGREES).move_to(ORIGIN).to_edge(RIGHT))

    def portrait_to_landscape(self):
        global_scene.play(self.animate.scale(1 / ls2pt).rotate(90 * DEGREES).move_to(ORIGIN))

    def load(self, state: MState):
        self.remove(self.s)
        self.s = state
        self.s.update_cs(self.cs)
        self.add(self.s)

    def transition(self, s2: MState, run_time=4):
        s2.update_cs(self.cs)
        global_scene.play(*self.s.get_animations(s2), run_time=run_time)

    def contextmanager_animation(self, submobject_getter, **kwargs):
        def wrapper():
            submobject = always_redraw(partial(submobject_getter, **kwargs))
            global_scene.add(submobject)
            global_scene.play(FadeIn(submobject), run_time=1)
            yield
            global_scene.play(FadeOut(submobject), run_time=1)
            global_scene.remove(submobject)
        ctx_mng = contextmanager(wrapper)()
        return ctx_mng

    def highlight(self, player_name):
        player = self.s.get_player(player_name)
        return self.contextmanager_animation(player.get_highlight)

    def marker_shadow(self):
        return self.contextmanager_animation(self.s.get_marking_shadow)

    def field_of_view(self, player_name):
        """
        :param scene: the scene to show the animations
        :param player_name: e.g. f'o1' for offense['1'] or d6 for defense['6']
        :return: contextmanager inside which the field of view is shown
        """
        player = self.s.get_player(player_name)
        return self.contextmanager_animation(player.get_field_of_view, field=self)

    def measure_distance(self, p1, p2, distance=None):
        p1, p2 = self.s.get_player(p1), self.s.get_player(p2)
        return self.contextmanager_animation(self.get_distance_measurement, p1=p1, p2=p2, distance=distance)

    def measure_angle(self, p1, p2, angle=None):
        p1, p2 = self.s.get_player(p1), self.s.get_player(p2)
        return self.contextmanager_animation(self.get_angle_measurement, p1=p1, p2=p2, angle=angle)

    def get_distance_measurement(self, p1, p2, distance=None):
        if distance is None:
            distance = np.linalg.norm(p1.position - p2.position)
        brace = BraceBetweenPoints(p1.get_center(), p2.get_center(), buff=1 * self.cs.current_scale()).set_z_index(3)
        brace.add(brace.get_text(rf'${int(round(distance))}$\,m').set_font_size(30).set_z_index(3))
        return brace

    def get_angle_measurement(self, p1, p2, angle=None):
        line = Line(p1.get_center(), p2.get_center())
        p1_x, p1_y = p1.position[:2]
        support_line_pos = Line(p1.get_center(), self.cs.c2p(cfg.field_width_m, p1_y))
        support_line_neg = Line(p1.get_center(), self.cs.c2p(0, p1_y))
        if angle is None:
            angle = (np.arctan2(*(p2.position - p1.position)[:2][::-1]) % TAU) / DEGREES
        if angle < 90:  # q1
            angle_to_line = angle
            line_1, line_2 = support_line_pos, line
        elif angle < 180:  # q2
            angle_to_line = 180 - angle
            line_1, line_2 = line, support_line_neg
        elif angle < 270:  # q3
            angle_to_line = angle - 180
            line_1, line_2 = support_line_neg, line
        else:  # q4
            angle_to_line = 360 - angle
            line_1, line_2 = line, support_line_pos
        angle_mob = Angle(line_1, line_2, radius=2.5 * self.cs.current_scale())
        tex = Tex(rf'${int(round(angle_to_line))}^\circ$').set_font_size(25).move_to(
            Angle(line_1, line_2, radius=2.5 * self.cs.current_scale() + MED_LARGE_BUFF).point_from_proportion(0.5))


        return VGroup(line_1, line_2, angle_mob, tex).set_z_index(3)





class FrameOfReference(Axes):
    def __init__(self, h, w, scale, *args, **kwargs):
        super().__init__(x_range=[0, w, w+1], y_range=[0, h, h+1], x_length=w*scale, y_length=h*scale, tips=False)
        self.rotate(180 * DEGREES)
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
        hightlight_opacity = 0.4
        c = Circle(
            radius=3 * self.frame_of_ref.current_scale(),
            fill_color=self.get_color(),
            stroke_width=0,
            fill_opacity=hightlight_opacity,
        ).shift(self.get_center() + IN)
        return c

    def get_field_of_view(self, field: Field):
        rect = field.rect.copy()
        reach = 50 * self.frame_of_ref.current_scale()
        opening_angle = 130 * DEGREES
        angle = self.frame_of_ref.get_angle() + self.angle + (PI - opening_angle) / 2
        field_of_view = AnnularSector(
            inner_radius=0,
            outer_radius=reach,
            angle=opening_angle,
            start_angle=angle,
        ).shift(self.get_center())
        gray_area = Difference(rect, field_of_view)
        gray_area.set_fill(BLACK, opacity=0.6).set_z_index(3).set_stroke(width=0)
        return gray_area





    @contextmanager
    def highlight(self):
        c = self.get_highlight()
        global_scene.play(FadeIn(c), run_time=0.8)
        self.add(c)
        yield
        global_scene.play(FadeOut(c), run_time=0.8)
        self.remove(c)


def get_minimum_rotation(radians):
    left = radians % TAU
    right = left - TAU
    return left if abs(left) < abs(right) else right


ls2pt = portrait_scale / landscape_scale

def delayed(func, delay):
    def wrapper(alpha):
        scale = 1 / (1 - delay)
        delayed_alpha = max([alpha - delay, 0]) * scale
        return func(delayed_alpha)
    return wrapper


class MoveDisc(Animation):
    # todo: calc delay and inflection based on the range and avg. speed of the throw
    # todo: eg: short throws move linearly while long thros slow down significantly
    def __init__(self, disc, target, delay, *args, **kwargs):
        super().__init__(disc, *args, **kwargs)
        self.target = target
        self.target_pos = self.target.get_center()
        self.start_pos = disc.get_center()
        self.rate_func = delayed(partial(rush_from, inflection=5), delay)

    def interpolate_mobject(self, alpha):
        k = self.rate_func(alpha)
        self.mobject.move_to((1-k) * self.start_pos + k * self.target_pos)





class MovePlayer(Animation):
    def __init__(self, mobject, target, cs: FrameOfReference, real_time=4, *args, **kwargs):
        super().__init__(mobject, *args, **kwargs)
        self.rate_func = partial(smooth, inflection=4)
        self.end_pose = target
        self.radians = get_minimum_rotation(target.angle - mobject.angle)
        self.cs = cs
        avg_speed_kmh = np.linalg.norm(target.position - mobject.position) / real_time * 3.6
        max_speed_floating = 50
        if avg_speed_kmh > max_speed_floating:
            self.interpolate_mobject = self.beam_interpolation  # todo: how to interpolate fast players?
        else:
            self.interpolate_mobject = self.float_interpolation

    # def beam_interpolation(self, alpha):
    #     self.mobject.become(self.starting_mobject if self.rate_func(alpha) < 0.5 else self.end_pose)

    def float_interpolation(self, alpha: float) -> None:
        self.mobject.become(self.starting_mobject)
        k = self.rate_func(alpha)
        self.mobject.angle = self.starting_mobject.angle + k * self.radians
        self.mobject.position = self.starting_mobject.position + k * (self.end_pose.position - self.starting_mobject.position)
        self.mobject.rotate(k * self.radians)
        pos = self.cs.c2p(*((1 - k) * self.starting_mobject.position) + k * self.end_pose.position)
        self.mobject.move_to(pos)

def create_movie(cls, debug, opengl=False, hq=False, output_file=None):
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
    def __init__(self, *args, **kwargs):
        config.pixel_width = 1800
        fh = config.frame_width
        config.frame_height = fh * 37 / 100
        config.pixel_height = round_to_odd(config.pixel_width * 37 / 100) + 1
        config.frame_height = config.frame_width * 37 / 100
        config.dry_run = True
        self.state = None

        super().__init__(*args, **kwargs)

    def get_img(self, state):
        if state is None:
            state = State.load('temp/s_copy.yaml')
        self.state = state
        self.clear()
        self.render()
        img = self.camera.get_image()
        return img

    def construct(self):
        s1 = MState(self.state)
        self.add(Field(state=s1, height=10, scale_for_landscape=False).scale(config.frame_width / 10).rotate(-90 * DEGREES))



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
        field = Field(self, state=self.states[0])
        for s1, s2 in zip(self.states, self.states[1:]):
            field.transition(s2)


def animation_from_state_dir(output_file=None):
    debug = False
    create_movie(DirAnimation, debug, output_file=output_file)





if __name__ == '__main__':
    # animation_from_state_dir(cfg.default_animation_file)
    StateImg.get()
    debug = False
    hq = True
    opengl = False
    # create_movie(scenes.MyScene, debug=debug, hq=hq, opengl=opengl)
    # create_movie(ultimate_scene.SingleTransition, debug=debug, hq=hq, opengl=opengl, play=True)
    # create_movie(LocalFrameOfReference)

