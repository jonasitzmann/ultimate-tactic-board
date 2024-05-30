from manim_animations import create_movie
from scenes import UltimateScene


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        self.write("Falls leicht möglich: Breakpass.", animate=False)
        self.write_small("Initiiert durch Werfer*in.", animate=False)
        self.wait()
        f.transition(s[1], run_time=1, d_delay=0.6, o_delay=0.3, disc_delay=0, linear_end=True)
        self.wait()


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
