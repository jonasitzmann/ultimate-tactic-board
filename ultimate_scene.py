from manim_animations import *
from manim import *
import cfg




class UltimateScene(Scene):
    def prepare(self, play_path=cfg.current_play_dir):
        states = get_states_from_dir(play_path)
        field = Field(self, states[0])
        return field, states

    def call(self, contextmanager):
        with contextmanager:
            pass

    @contextmanager
    def tex(self, tex):
        mob = Tex(tex).to_corner(UL)
        self.play(Write(mob))
        self.add(mob)
        yield
        self.play(FadeOut(mob, run_time=0.5))
        self.remove(mob)



class MyScene(UltimateScene):
    def construct(self):
        f, (s1, s2) = self.prepare('example_play')
        with self.tex('measure distances'), f.measure_distance('o2', 'o4'):
            self.wait(2)
        with self.tex('measure angles'), f.measure_angle('o2', 'o3'):
            self.wait(2)
        with self.tex('highlight players:'):
            [self.call(f.highlight(p)) for p in 'o1 o4 d2 d5'.split()]
        with self.tex('add marker shadow'), f.marker_shadow():
            self.wait(2)
        with self.tex("show player's field of view"), f.field_of_view('o1'):
            self.wait(2)
        tex = 'animate everything'
        with self.tex(tex), f.highlight('o4'), f.marker_shadow(), f.field_of_view('o1'),\
                f.measure_distance('o4', 'd4'), f.measure_angle('o2', 'o3'):
            self.wait()
            f.transition(s2)
            self.wait(2)
