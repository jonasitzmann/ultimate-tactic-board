from manim_animations import create_movie
from scenes import UltimateScene
from manim import *
from manim_presentation import Slide


class Play1(UltimateScene, Slide):
    def pplay(self, *args, **kwargs):
        self.play(*args, **kwargs)
        self.pause()

    def construct(self):
        plan_a = Tex("Plan A").to_edge(UP, MED_SMALL_BUFF).shift(2*RIGHT)
        plan_b = Tex("Plan B").to_corner(UR, MED_SMALL_BUFF).shift(LEFT)
        tex = Tex(self.get_tex('slide_1')).to_corner(UL, MED_SMALL_BUFF)
        self.wait(0.1)
        self.pause()
        self.pplay(Write(tex))
        self.play(Write(plan_a))
        f_1, s = self.prepare()
        f_1.landscape_to_portrait(animate=False).scale(0.9).next_to(plan_a, DOWN, MED_SMALL_BUFF)
        self.play(FadeIn(f_1))
        plan_a_arrows = f_1.get_arrows(s[0], s[1])
        f_1.add(plan_a_arrows)
        self.play(Write(plan_a_arrows))
        self.add(plan_a_arrows)
        self.pause()
        f_1.transition(s[1], run_time=3, disc_delay=0.2)
        self.pause()
        f2 = f_1.copy().next_to(plan_b, DOWN)
        self.play(Write(plan_b))
        self.play(FadeIn(f2))
        plan_b_arrows = f2.get_arrows(s[2], s[3])
        self.add(plan_b_arrows)
        self.pause()
        f2.transition(s[0], run_time=0.3)
        self.play(Write(plan_b_arrows))
        f2.add(plan_b_arrows)
        self.pause()
        f2.transition(s[2], run_time=3), f2.transition(s[3], run_time=3)
        self.pause()
        l1 = Line(f_1.get_corner(UL), f2.get_corner(DR), color=RED_D, stroke_width=10, z_index=5)
        l2 = Line(f_1.get_corner(DL), f2.get_corner(UR), color=RED_D, stroke_width=10, z_index=5)
        self.pplay(DrawBorderThenFill(l1), DrawBorderThenFill(l2))
        self.wait()


def render_scene():
    # create_movie(Play1, debug=False, hq=True, output_file='play1.mp4')
    bin_dir = '/home/jonas/.conda/envs/tactics_board/bin'
    os.system(f'{bin_dir}/manim-presentation Play1 --fullscreen')


if __name__ == '__main__':
    render_scene()
