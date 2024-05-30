from manim import *
from manim_animations import create_movie
from scenes import UltimateScene
import numpy as np


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        f.load_state(s[5])


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        self.write(
            "An der Sideline im Flow wieder in die Mitte spielen (geschlossene Seite)",
            animate=False,
        )
        self.write_small("Ganzer Durchlauf", animate=False)
        self.wait()
        f.transition(s[1], run_time=0.7)
        f.transition(s[2], run_time=0.7)
        f.transition(s[3], run_time=0.7)
        f.transition(s[4], run_time=0.7)
        f.transition(s[5], run_time=0.7)
        f.transition(s[6], run_time=0.7)
        f.transition(s[7], run_time=0.7)
        f.transition(s[8], run_time=0.7)
        self.wait(2)


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
