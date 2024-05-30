from manim_animations import create_movie
from scenes import UltimateScene


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        f.load_state(s[0])


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        f.load_state(s[0])
        self.write("Clearing per Überläufer", animate=False)
        self.wait()
        f.transition(s[1], run_time=0.5)
        f.fake("o2")
        f.transition(s[2], run_time=0.7)
        f.transition(s[3], run_time=0.7, linear_end=True)
        f.fake("o2", run_time=0.1)
        f.transition(s[4], run_time=0.5, d_delay=0, linear_start=True, linear_end=True)
        f.transition(s[5], run_time=0.7, linear_start=True, linear_end=True, d_delay=0)
        self.wait(2)


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
