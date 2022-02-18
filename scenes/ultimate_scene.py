import sys
from manim import *
from contextlib import contextmanager, ExitStack
import cfg


@contextmanager
def multi_context(contexts):  # https://stackoverflow.com/a/45681273
    with ExitStack() as stack:
        yield [stack.enter_context(c) for c in contexts]


class UltimateScene(Scene):
    def prepare(self, play_path='.'):
        from manim_animations import Field, get_states_from_dir
        states = get_states_from_dir(play_path)
        field = Field(self, states[0])
        # self.logo = ImageMobject('../../' + cfg.logo_path).scale(0.25).to_corner(DR, buff=SMALL_BUFF).set_opacity(0.3)
        # self.add(self.logo)
        return field, states

    def call(self, *contextmanagers, t=0):
        with multi_context(contextmanagers):
            self.wait(t)

    @contextmanager
    def tex(self, tex):
        mob = Tex(tex).to_corner(UL)
        self.play(Write(mob))
        self.add(mob)
        yield
        self.play(FadeOut(mob, run_time=0.5))
        self.remove(mob)
