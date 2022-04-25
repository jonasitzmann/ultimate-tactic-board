from manim_animations import create_movie
from scenes import UltimateScene
from manim import *
from manim_presentation import Slide


class Play1(UltimateScene, Slide):
    def pplay(self, *args, **kwargs):
        self.play(*args, **kwargs)
        self.pause()

    def construct(self):
        self.wait(0.1)
        self.pause()
        header = Tex('Playbook').to_corner(UL)
        self.pplay(Write(header))
        cross = Cross(header, WHITE, stroke_width=4)
        self.wait()
        self.play(Write(cross))
        self.wait()
        new_header = Tex('Offense nimmt, was Defense anbietet').to_corner(UL)
        self.pplay(FadeOut(cross), Transform(header, new_header))
        f, s = self.prepare()
        self.wait()
        description = Tex(r"Offene Pässe + Give'n Go").to_corner(DL)
        self.pplay(Write(description))
        f.transition(s[1], run_time=2, disc_delay=0)
        f.transition(s[2], run_time=2, disc_delay=0.4)
        self.pause()
        self.play(FadeOut(description))


        f.load_state(s[0])
        description = Tex(r"Einfache Breaks").to_corner(DL)
        f.transition(s[3], run_time=1)
        self.pplay(Write(description))
        f.transition(s[4], run_time=2, disc_delay=0.6)
        f.transition(s[5], run_time=2, disc_delay=0.1)
        self.pause()
        self.play(FadeOut(description))

        f.load_state(s[3])
        description = Tex(r"Poaches").to_corner(DL)
        f.transition(s[6], run_time=1)
        self.pplay(Write(description))
        with f.measure_distance('o3', 'd3'):
            f.transition(s[7], run_time=2, disc_delay=0.1)
        self.pause()
        self.play(FadeOut(description))

        f.load_state(s[6])
        description = Tex(r"Face Mark").to_corner(DL)
        f.transition(s[8], run_time=1)
        self.pplay(Write(description))
        with f.field_of_view('d2'):
            f.transition(s[9], run_time=2, disc_delay=0.6)
            f.transition(s[10], run_time=2, disc_delay=0.1)
        self.pause()
        self.wait()


def render_scene():
    # create_movie(Play1, debug=True, hq=True, output_file='play1.mp4')
    bin_dir = '/home/jonas/.conda/envs/tactics_board/bin'
    os.system(f'{bin_dir}/manim-presentation Play1 --fullscreen')


if __name__ == '__main__':
    render_scene()
