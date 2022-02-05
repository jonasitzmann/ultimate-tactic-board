from manim import *
import inspect
import os
from state import State
from functools import partial
landscape_scale = (config['frame_x_radius'] - DEFAULT_MOBJECT_TO_EDGE_BUFFER) / 50  # landscape
portrait_scale = (config['frame_y_radius'] - DEFAULT_MOBJECT_TO_EDGE_BUFFER) / 50  # portrait

myred = '#9c360d'
myblue = '#3a32ff'

class MState(VGroup):
    def __init__(self, state: State, cs=None):
        self.cs = cs
        self.state = state or State()
        self.offense, self.defense = VDict(), VDict()
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

    def get_animations(self, new_state):
        movements = []
        for current, target in [(self.offense, new_state.offense), (self.defense, new_state.defense)]:
            for k, v in current.submob_dict.items():
                t = target.submob_dict.get(k, None)
                if t is not None:
                    movements.append(MovePlayer(v, t, self.cs))
        return movements


class Field(VGroup):
    def __init__(self, state=None, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.cs = FrameOfReference(100, 37, landscape_scale)
        self.add(self.cs)
        self.state = None
        h, w = np.array([100, 37]) * landscape_scale
        ez_height = 18 * landscape_scale
        self.add(Rectangle(height=h, width=w, fill_color='#728669', fill_opacity=0.7))
        for direction in [UP, DOWN]:
            self.add(Line(ORIGIN, w * RIGHT).shift(w / 2 * LEFT + (h / 2 - ez_height) * direction))
            self.add(Cross(stroke_color=WHITE, scale_factor=0.05, stroke_width=3).shift((h/2 - 2*ez_height) * direction))
        if state is not None:
            self.load(state)

    def load(self, state: MState):
        self.remove(self.state)
        self.state = state
        self.state.update_cs(self.cs)
        self.add(self.state)

    def transition(self, s1: MState, s2: MState):
        self.load(s1)
        s2.update_cs(self.cs)
        return s1.get_animations(s2)


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


class MPlayer(Ellipse):
    def __init__(self, angle, position, frame_of_ref: FrameOfReference, color=None, role='o'):
        size = np.array([1.5, 1.5] if role == 'o' else [2.3, 0.5]) * frame_of_ref.current_scale()
        self.role = role
        if color is None:
            color = myred if self.is_o else myblue
        super().__init__(*size, color=color, fill_color=color, fill_opacity=1)
        self.angle = angle
        self.position = position
        self.frame_of_ref = frame_of_ref
        self.nose = Triangle(color=color).scale_to_fit_height(0.5*frame_of_ref.current_scale()).move_to(size[1]/2*UP, aligned_edge=DOWN)
        self.add(self.nose)
        self.set_pose(position, angle)
    @property
    def is_o(self):
        return self.role == 'o'

    def set_pose(self, position, angle):
        self.rotate(angle + self.frame_of_ref.get_angle())
        self.move_to(self.frame_of_ref.c2p(*position))


def get_minimum_rotation(radians):
    left = radians % TAU
    right = left - TAU
    return left if abs(left) < abs(right) else right


ls2pt = portrait_scale / landscape_scale

class UltimateScene(Scene):
    def construct(self):
        self.play = partial(self.play, run_time=3)
        s1, s2 = MState(State.load('temp/s.yaml')), MState(State.load('temp/s_2.yaml'))
        tex = Tex('Die \glqq Nasen\grqq~ zeigen die Blickrichtung der Spieler').to_corner(UP + LEFT)
        field = Field(s1).rotate(90*DEGREES).next_to(tex, DOWN, aligned_edge=LEFT)
        self.play(Write(tex), DrawBorderThenFill(field))
        self.play(*field.transition(s1, s2))
        # self.play(FadeOut(tex), field.animate.scale(ls2pt).rotate(90 * DEGREES).move_to(ORIGIN).to_edge(RIGHT))
        # tex = Tex(r"""
        # Das Spielfeld kann auch hochkant angezeigt werden\\
        # um Platz für lange Erläuterungen zu haben.
        # \begin{itemize}
        #  \item Die Spieler entscheiden jetzt selbst,\\
        #  ob sie nach rechts oder links drehen
        #  \item Allerdings sind ihre Bewegungen insgesamt noch unrealistisch,\\
        #  weil sie nicht in Blickrichtung laufen.
        #  \item Irgendwann sieht man hier auch ne Scheibe
        #  \item Auch hochkant funktionieren alle Bewegungen
        # \end{itemize}
        # """, font_size=35).to_edge(LEFT)
        # self.play(Write(tex))
        # self.wait(7)
        # self.play(*field.transition(s2, s1))
        # self.wait(2)
        # self.play(FadeOut(tex, run_time=1), field.animate.move_to(ORIGIN).rotate(-90*DEGREES).scale(1 / ls2pt))
        self.wait(3)


class MovePlayer(Animation):
    def __init__(self, mobject, target, cs: FrameOfReference, real_time=4, *args, **kwargs):
        super().__init__(mobject, *args, **kwargs)
        self.end_pose = target
        self.radians = get_minimum_rotation(target.angle - mobject.angle)
        self.cs = cs
        avg_speed_kmh = np.linalg.norm(target.position - mobject.position) / real_time * 3.6
        max_speed_floating = 10
        self
        if avg_speed_kmh > max_speed_floating:
            self.interpolate_mobject = self.beam_interpolation  # todo: how to interpolate fast players?
        else:
            self.interpolate_mobject = self.float_interpolation

    def beam_interpolation(self, alpha):
        self.mobject.become(self.starting_mobject if self.rate_func(alpha) < 0.5 else self.end_pose)

    def float_interpolation(self, alpha: float) -> None:
        self.mobject.become(self.starting_mobject)
        k = self.rate_func(alpha)
        self.mobject.rotate(k * self.radians)
        pos = self.cs.c2p(*((1 - k) * self.mobject.position) + k * self.end_pose.position)
        self.mobject.move_to(pos)

def create_movie(cls, debug=False, hq=False):
    if debug:
        cls().render(True)
    else:
        cls_name, cls_path = cls.__name__, inspect.getfile(cls)
        os.system(f'~/.conda/envs/tactics_board/bin/manim -pq{"h" if hq else "l"} {cls_path} {cls_name}')

if __name__ == '__main__':
    debug = False
    hq = False
    create_movie(UltimateScene, debug, hq)
    # create_movie(LocalFrameOfReference)

