import sys
from contextlib import contextmanager
from functools import partial
import copy
from manim.renderer.opengl_renderer import OpenGLRenderer

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


def get_disc(state_disc, cs):
    group = VGroup()
    if len(state_disc) > 0:
        group.add(Dot(stroke_width=0, fill_color=WHITE, fill_opacity=1, radius=0.5 * cs.scale_(), z_index=3))
        group.move_to(cs.c2p(*state_disc))
    return group



global_scene = None

class Field(VGroup):
    global scene
    def __init__(self, scene=None, state=None, height=10, scale_for_landscape=True, **kwargs):
        VGroup.__init__(self, **kwargs)
        scale = height / 100
        self.cs = FrameOfReference(100, 37, scale)
        h, w = np.array([100, 37]) * scale
        ez_height = 18 * scale
        self.background = Rectangle(height=h, width=w, fill_color='#728669', fill_opacity=0.75)
        for direction, color in zip([UP, DOWN], [myred, myblue]):
            self.background.add(Line(ORIGIN, w * RIGHT, color=color).shift(w / 2 * LEFT + (h / 2 - ez_height) * direction))
            self.background.add(Cross(stroke_color=WHITE, scale_factor=0.05, stroke_width=3).shift((h/2 - 2*ez_height) * direction))
        self.bounding_box = Rectangle(height=h, width=w)
        self.background.set_z_index(-1)
        self.state = state
        self.players = self.disc = None
        super().__init__(self.cs, self.bounding_box, self.background)
        self.load_state(state)
        global global_scene
        global_scene = scene
        if global_scene is not None:
            global_scene.add(self)
        if scale_for_landscape:
            self.scale((config.frame_x_radius - DEFAULT_MOBJECT_TO_EDGE_BUFFER) / 5).rotate(-90 * DEGREES)

    def load_state(self, state):
        self.state = copy.deepcopy(state)
        new_disc = get_disc(self.state.disc, self.cs)
        new_players = VGroup(*[MPlayer(p, self.cs) for p in self.state.playerlist])
        if self.players is None:
            self.players = new_players
            self.add(self.players)
        else:
            self.players.become(new_players)
        if self.disc is None:
            self.disc = new_disc
            self.add(self.disc)
        else:
            self.disc.become(new_disc)

    def landscape_to_portrait(self):
        global_scene.play(self.animate.scale(ls2pt).rotate(-90 * DEGREES).move_to(ORIGIN).to_edge(RIGHT))

    def portrait_to_landscape(self):
        global_scene.play(self.animate.scale(1 / ls2pt).rotate(90 * DEGREES).move_to(ORIGIN))

    def transition(self, s2: State, run_time=4):
        global_scene.play(*self.get_animations(s2), run_time=run_time)
        self.load_state(s2)

    def contextmanager_animation(self, submobject_getter, redraw=True, fade=True, **kwargs):
        def wrapper():
            if redraw:
                submobject = always_redraw(partial(submobject_getter, **kwargs))
            else:
                submobject = submobject_getter(**kwargs)
            global_scene.add(submobject)
            if fade:
                global_scene.play(FadeIn(submobject), run_time=1)
            yield submobject
            if fade:
                global_scene.play(FadeOut(submobject), run_time=1)
            global_scene.remove(submobject)
        ctx_mng = contextmanager(wrapper)()
        return ctx_mng

    def highlight(self, player_name, **kwargs):
        player = self.get_player(player_name)
        return self.contextmanager_animation(player.get_highlight, redraw=False, **kwargs)

    def marker_shadow(self):
        return self.contextmanager_animation(self.get_marking_shadow)

    def field_of_view(self, player_name, *args, **kwargs):
        """
        :param scene: the scene to show the animations
        :param player_name: e.g. f'o1' for offense['1'] or d6 for defense['6']
        :return: contextmanager inside which the field of view is shown
        """
        player = self.get_player(player_name)
        return self.contextmanager_animation(player.get_field_of_view, field=self, *args, **kwargs)

    def measure_distance(self, p1, p2, distance=None, *args, **kwargs):
        p1, p2 = self.get_player(p1), self.get_player(p2)
        return self.contextmanager_animation(self.get_distance_measurement, redraw=False, p1=p1, p2=p2, distance=distance, *args, **kwargs)

    def measure_angle(self, p1, p2, angle=None, **kwargs):
        p1, p2 = self.get_player(p1), self.get_player(p2)
        return self.contextmanager_animation(self.get_angle_measurement, redraw=False, p1=p1, p2=p2, angle=angle, **kwargs)

    def get_distance_measurement(self, p1, p2, distance=None):
        color = GRAY_B
        number, unit = label = VGroup(
            DecimalNumber(
                num_decimal_places=0,
                font_size=20, color=color
            ).set_z_index(3),
            Tex(r'm', font_size=20, z_index=3, color=color),
        )
        def get_distance(distance, p1, p2):
            if distance is None:
                distance = np.linalg.norm(p1.player.pos - p2.player.pos)
            return int(round(distance))

        def get_brace(p1, p2):
            brace = BraceBetweenPoints(p1.get_center(), p2.get_center(), buff=1 * self.cs.scale_(), color=color)
            brace.set_z_index(0)
            number.set_value(get_distance(distance, p1, p2))
            label.arrange(RIGHT, buff=0.5 * SMALL_BUFF, aligned_edge=DOWN)
            brace.put_at_tip(label, use_next_to=False, buff=SMALL_BUFF)
            label.set_z_index(3)
            return brace
        brace = always_redraw(partial(get_brace, p1, p2))
        return VGroup(brace, label)

    def get_angle_measurement(self, p1, p2, angle=None):
        color = GRAY_B
        number, unit = label = VGroup(
            DecimalNumber(
                num_decimal_places=0,
                font_size=20, color=color
            ),
            Tex(r'$^\circ$', font_size=20, z_index=3, color=color)
        )

        def get_lines_and_arc(p1, p2, angle):
            line = Line(p1.get_center(), p2.get_center(), color=color)
            p1_x, p1_y = p1.player.pos
            support_line_pos = Line(p1.get_center(), self.cs.c2p(cfg.field_width_m, p1_y), color=color)
            support_line_neg = Line(p1.get_center(), self.cs.c2p(0, p1_y), color=color)
            if angle is None:
                angle = (np.arctan2(*(p2.player.pos - p1.player.pos)[:2][::-1]) + PI % TAU) / DEGREES
            if angle < 90:  # q1
                angle_to_line = angle
                line_2, line_1 = support_line_pos, line
            elif angle < 180:  # q2
                angle_to_line = 180 - angle
                line_2, line_1 = line, support_line_neg
            elif angle < 270:  # q3
                angle_to_line = angle - 180
                line_2, line_1 = support_line_neg, line
            else:  # q4
                angle_to_line = 360 - angle
                line_2, line_1 = line, support_line_pos
            angle_mob = Angle(line_1, line_2, radius=2.5 * self.cs.scale_(), color=color)
            angle_mob.add(line_1, line_2)
            angle_mob.set_z_index(0)
            number.set_value(angle_to_line)
            return angle_mob
        angle_mob = always_redraw(lambda: get_lines_and_arc(p1, p2, angle))

        label.add_updater(lambda x: label.arrange(RIGHT, buff=0.5*SMALL_BUFF, aligned_edge=DOWN))
        def move_label(label, angle_mob):
            lines = [[l.get_start(), l.get_end()] for l in angle_mob.get_lines()]
            c = line_intersection(*lines)
            direction = angle_mob.point_from_proportion(0.5) - c
            unit_vec = direction / np.linalg.norm(direction)
            label.arrange(RIGHT, buff=0.3 * SMALL_BUFF, aligned_edge=UP)
            label.move_to(c + 1.3 * MED_LARGE_BUFF * unit_vec)
        label.add_updater(lambda m: move_label(m, angle_mob))
        return VGroup(angle_mob, label)

    # former MState methods
    @contextmanager
    def marker_shadow(self):
        shadow = always_redraw(self.get_marking_shadow)
        global_scene.add(shadow)
        global_scene.play(FadeIn(shadow), run_time=0.8)
        yield
        global_scene.play(FadeOut(shadow), run_time=0.8)
        global_scene.remove(shadow)

    def get_marking_shadow(self):
        marking_reach = 2 * self.cs.scale_()
        offender = sorted(self.offenders(), key=lambda x: np.linalg.norm(x.player.pos - self.state.disc))[0]
        marker = sorted(self.defenders(), key=lambda x: np.linalg.norm(x.player.pos - offender.player.pos))[0]
        o_pos, d_pos = offender.get_center(), marker.get_center()
        marking_shadow_size = 30 * self.cs.scale_()
        marking_line = Line(d_pos + 0.5 * marking_reach * LEFT, d_pos + 0.5 * marking_reach * RIGHT)
        marking_line.rotate(marker.player.manimangle + self.cs.get_angle())
        area = get_trapezoid(o_pos, marking_line.get_start(), marking_line.get_end(), marking_shadow_size)
        area = Intersection(area, self.cs.rect, stroke_width=0, fill_color=marker.get_color(), fill_opacity=0.3)
        area.z_index = 1
        return area

    def get_player(self, player_name):
        role, label = player_name[0], player_name[1:]
        return sorted(self.players, key=lambda x: x.player.label == label and x.player.role == role)[-1]

    def offenders(self):
        return VGroup(*[p for p in self.players if p.is_o])

    def defenders(self):
        return VGroup(*[p for p in self.players if not p.is_o])

    def get_animations(self, new_state, disc_delay=0.5):
        movements = []
        for player in self.players:
            movements.append(MovePlayer(player, new_state))
        movements.append(MoveDisc(self.disc, get_disc(new_state.disc, self.cs), disc_delay))
        return movements





class FrameOfReference(Axes):
    def __init__(self, h, w, scale, *args, **kwargs):
        super().__init__(x_range=[0, w, w+1], y_range=[0, h, h+1], x_length=w*scale, y_length=h*scale, tips=False)
        self.rotate(180 * DEGREES)
        self.c2p = self.coords_to_point

    def scale_(self):
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
    def __init__(self, player, frame_of_ref):
        self.player = player
        size = np.array([1.2, 1.2] if player.role == 'o' else [2, 0.5]) * frame_of_ref.scale_()
        color = myblue if self.is_o else myred
        super().__init__(color=color)
        self.ellipse=Ellipse(*size, stroke_width=0, fill_color=color, fill_opacity=1, z_index=2)
        self.cs = frame_of_ref
        self.nose = Triangle(fill_color=color, fill_opacity=1, z_index=2, stroke_width=0).scale_to_fit_height(0.5 * frame_of_ref.scale_())
        self.nose.next_to(self.ellipse, UP, buff=-size[1]/5)
        self.add(self.ellipse, self.nose)
        self.rotate(self.player.manimangle + self.cs.get_angle())
        self.move_to(self.cs.c2p(*self.player.manimpos))

    @property
    def is_o(self):
        return self.player.role == 'o'


    def get_highlight(self):
        hightlight_opacity = 0.4
        c = Circle(
            radius=3 * self.cs.scale_(),
            fill_color=self.get_color(),
            stroke_width=0,
            fill_opacity=hightlight_opacity,
        )
        moveto = lambda c: c.move_to(self.get_center())
        moveto(c)
        c.add_updater(moveto)
        return c

    def get_field_of_view(self, field: Field):
        reach = 50 * self.cs.scale_()
        opening_angle = 130 * DEGREES
        angle = self.cs.get_angle() + self.player.manimangle + (PI - opening_angle) / 2
        field_of_view = AnnularSector(
            inner_radius=0,
            outer_radius=reach,
            angle=opening_angle,
            start_angle=angle,
        ).shift(self.get_center())
        gray_area = Difference(field.bounding_box, field_of_view)
        gray_area.set_fill(BLACK, opacity=0.6).set_stroke(width=0)
        gray_area.set_z_index(3)
        return gray_area





    @contextmanager
    def highlight(self):
        c = self.get_highlight()
        global_scene.play(FadeIn(c), run_time=0.8)
        self.add(c)
        yield
        global_scene.play(FadeOut(c), run_time=0.8)
        self.remove(c)


def get_minimum_rotation(degrees):
    left = degrees % 360
    right = left - 360
    return left if abs(left) < abs(right) else right


ls2pt = portrait_scale / landscape_scale

def delayed(func, delay):
    def wrapper(alpha):
        scale = 1 / (1 - delay)
        delayed_alpha = max([alpha - delay, 0]) * scale
        return func(delayed_alpha)
    return wrapper


def compressed_front(func, neg_delay):
    def wrapper(alpha):
        scale = 1 / (1 - neg_delay)
        scaled_alpha = min([alpha * scale, 1])
        return func(scaled_alpha)
    return wrapper



class MoveDisc(Animation):
    # todo: calc delay and inflection based on the range and avg. speed of the throw
    # todo: eg: short throws move linearly while long throws slow down significantly
    def __init__(self, disc, target, delay, *args, **kwargs):
        super().__init__(disc, *args, **kwargs)
        self.target = target
        self.target_pos = self.target.get_center()
        self.start_pos = disc.get_center()
        self.rate_func = delayed(partial(rush_from, inflection=5), delay)
        self.total_shift = self.target_pos - self.start_pos
        self.current_k = 0

    def interpolate_mobject(self, alpha):
        k = self.rate_func(alpha)
        delta_k = k - self.current_k
        self.mobject.shift(self.total_shift * delta_k)
        self.current_k = k





class MovePlayer(Animation):
    def __init__(self, mobject, end_state: State, real_time=4, *args, **kwargs):
        super().__init__(mobject, *args, **kwargs)
        self.rate_func = partial(smooth, inflection=4)
        self.start_state = copy.deepcopy(mobject.player)
        self.end_state = end_state.get_player(mobject.player)
        self.cs = mobject.cs
        self.total_rotation = get_minimum_rotation(self.end_state.angle - self.start_state.angle)
        self.total_rotation_rad = self.total_rotation * DEGREES
        avg_speed_kmh = np.linalg.norm(self.start_state.pos - self.end_state.pos) / real_time * 3.6
        max_speed_floating = 7
        self.last_k_rot = 0
        self.total_shift = self.cs.c2p(*self.end_state.manimpos) - self.cs.c2p(*self.start_state.manimpos)
        self.rate_func_compressed = compressed_front(self.rate_func, 0.7)
        if avg_speed_kmh > max_speed_floating:
            self.interpolate_mobject = self.run_interpolation
        else:
            self.interpolate_mobject = self.float_interpolation

    def interpolation(self, k_rot, k_pos):
        delta_k_rot = k_rot - self.last_k_rot
        d_angle = self.total_rotation_rad * delta_k_rot
        self.mobject.player.angle = self.start_state.angle + k_rot * self.total_rotation
        self.mobject.player.pos = (1 - k_pos) * self.start_state.pos + k_pos * self.end_state.pos
        self.mobject.move_to(self.mobject.cs.c2p(*self.mobject.player.manimpos))
        self.mobject.rotate(d_angle)
        self.last_k_rot = k_rot

    def float_interpolation(self, alpha):
        k = self.rate_func(alpha)
        return self.interpolation(k, k)

    def run_interpolation(self, alpha):
        k_pos = self.rate_func(alpha)
        k_rot = self.rate_func_compressed(alpha)
        return self.interpolation(k_rot, k_pos)





def create_movie(cls, debug, opengl=False, hq=False, output_file=None, dry_run=False):
    if debug:
        obj = cls()
        obj.render(True)
    else:
        cls_name, cls_path = cls.__name__, inspect.getfile(cls)
        opengl_part = '--renderer=opengl' if opengl else ''
        output_file_ = cfg.default_animation_file if output_file is None else output_file
        ouput_part = ('-o ../../../../' + output_file_) if output_file else ''
        dry_run_part = '--dry_run' if dry_run else ''
        command = f'{opengl_part} {ouput_part} -pq{"h" if hq else "l"} {dry_run_part} {cls_path} {cls_name}'
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
        self.annotations = []
        self.state = None

        super().__init__(*args, **kwargs)

    def get_img(self, state, annotations=None):
        self.state = State.load('temp/s_copy.yaml') if state is None else state
        if annotations is not None:
            self.annotations = annotations
        self.clear()
        self.render()
        img = self.camera.get_image()
        return img

    def construct(self):
        field = Field(self, state=self.state, height=10, scale_for_landscape=False).scale(config.frame_width / 10).rotate(-90 * DEGREES)
        for annotation in self.annotations:
            annotation(field, fade=False).__enter__()



def get_states_from_dir(dir_path):
    sort_func = lambda filepath: f'{int(filepath.split("/")[-1].split(".")[0]):02d}'
    states = [State.load(f) for f in sorted([f for f in glob(f'{dir_path}/*.yaml')], key=sort_func)]
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
    # StateImg.get()
    debug = False
    hq = True
    opengl = False
    # create_movie(ultimate_scene.SingleTransition, debug=debug, hq=hq, opengl=opengl, play=True)
    # create_movie(LocalFrameOfReference)

