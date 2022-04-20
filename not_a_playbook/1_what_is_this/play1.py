from manim_animations import create_movie
from scenes import UltimateScene
from manim import *


class Play1(UltimateScene):
    def construct(self):
        header = Tex('Playbook').to_corner(UL)
        self.play(Write(header))
        self.wait()
        cross = Cross(header, WHITE, stroke_width=4)
        self.wait()
        self.play(Write(cross))
        self.wait()
        new_header = Tex('Offense nimmt, was Defense anbietet').to_corner(UL)
        self.play(FadeOut(cross), Transform(header, new_header))
        self.wait()
        f, s = self.prepare()
        self.wait()
        description = Tex(r"Offene Pässe + Give'n Go").to_corner(DL)
        self.play(Write(description))
        self.wait()
        f.transition(s[1], run_time=2, disc_delay=0)
        f.transition(s[2], run_time=2, disc_delay=0.4)
        self.wait(1)
        self.play(FadeOut(description))


        f.load_state(s[0])
        description = Tex(r"Einfache Breaks").to_corner(DL)
        f.transition(s[3], run_time=1)
        self.play(Write(description))
        self.wait()
        f.transition(s[4], run_time=2, disc_delay=0.6)
        f.transition(s[5], run_time=2, disc_delay=0.1)
        self.wait(2)
        self.play(FadeOut(description))

        f.load_state(s[3])
        description = Tex(r"Poaches").to_corner(DL)
        f.transition(s[6], run_time=1)
        self.play(Write(description))
        self.wait()
        with f.measure_distance('o3', 'd3'):
            f.transition(s[7], run_time=2, disc_delay=0.1)
        self.wait()
        self.play(FadeOut(description))

        f.load_state(s[6])
        description = Tex(r"Face Mark").to_corner(DL)
        f.transition(s[8], run_time=1)
        self.play(Write(description))
        self.wait()
        with f.field_of_view('d2'):
            f.transition(s[9], run_time=2, disc_delay=0.6)
            f.transition(s[10], run_time=2, disc_delay=0.1)
        self.wait()
        self.play(FadeOut(description))

def render_scene():
    create_movie(Play1, debug=True, hq=True, output_file='play1.mp4')


if __name__ == '__main__':
    render_scene()
