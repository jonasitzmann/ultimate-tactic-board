from manim_animations import create_movie
from scenes import UltimateScene


class 1(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        f.transition(s[1], run_time=2)
        f.transition(s[2], run_time=2)
        f.transition(s[3], run_time=2)
        f.transition(s[4], run_time=2)
        f.transition(s[5], run_time=2)
        f.transition(s[6], run_time=2)
        f.transition(s[7], run_time=2)


def render_scene():
    create_movie(1, debug=False, hq=True, output_file='1.mp4')


if __name__ == '__main__':
    render_scene()
