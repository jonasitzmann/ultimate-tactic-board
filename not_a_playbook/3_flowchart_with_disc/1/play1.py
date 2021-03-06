from manim_animations import create_movie
from scenes import UltimateScene


class Play1(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        f.transition(s[1], run_time=1)
        f.fake('o2')
        f.transition(s[2], run_time=2, d_delay=0.3)
        with f.field_of_view('o2'):
            self.wait(2)
        f.transition(s[3], run_time=2, disc_delay=0, d_delay=0.3, linear_end=True)
        f.transition(s[4], run_time=1, disc_delay=0, linear_start=True)


def render_scene():
    create_movie(Play1, debug=False, hq=True, output_file='play1.mp4')


if __name__ == '__main__':
    render_scene()
