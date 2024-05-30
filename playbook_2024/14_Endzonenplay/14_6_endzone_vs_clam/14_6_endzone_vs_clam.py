from manim_animations import create_movie
from scenes import UltimateScene


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        self.wait()
        # f.transition(s[1], run_time=1, d_delay=0.4, o_delay=0.2, disc_delay=0)


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
