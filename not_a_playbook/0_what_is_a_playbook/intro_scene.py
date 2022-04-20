from manim_animations import create_movie
from scenes import UltimateScene
from manim import *


class Play1(UltimateScene):
    def construct(self):
        plan_a = Tex("Plan A").to_edge(UP).shift(2*RIGHT)
        plan_b = Tex("Plan B").to_corner(UR).shift(LEFT)
        tex = Tex(self.get_tex('slide_1')).to_corner(UL)
        self.play(Write(tex))
        self.wait(2)
        self.play(Write(plan_a))
        f_1, s = self.prepare()
        f_1.landscape_to_portrait(animate=False).scale(0.95).next_to(plan_a, DOWN)
        self.play(FadeIn(f_1))
        plan_a_arrows = f_1.get_arrows(s[0], s[1])
        f_1.add(plan_a_arrows)
        self.play(Write(plan_a_arrows))
        f_1.transition(s[1], run_time=3, disc_delay=0.2)
        self.wait(2)
        f2 = f_1.copy().next_to(plan_b, DOWN)
        self.play(Write(plan_b))
        self.play(FadeIn(f2))
        plan_b_arrows = f2.get_arrows(s[2], s[3])
        self.wait(1)
        f2.transition(s[0], run_time=0.3)
        self.play(Write(plan_b_arrows))
        self.wait(1)
        f2.add(plan_b_arrows)
        self.wait(1)
        f2.transition(s[2], run_time=3), f2.transition(s[3], run_time=3)
        self.wait(2)
        l1 = Line(f_1.get_corner(UL), f2.get_corner(DR), color=RED_D, stroke_width=10, z_index=5)
        l2 = Line(f_1.get_corner(DL), f2.get_corner(UR), color=RED_D, stroke_width=10, z_index=5)
        self.play(DrawBorderThenFill(l1), DrawBorderThenFill(l2))
        self.wait(1)
        self.play(FadeOut(l1), FadeOut(l2), FadeOut(f_1), FadeOut(f2), FadeOut(plan_a), FadeOut(plan_b), FadeOut(tex))
        self.remove(f_1, f2)



def render_scene():
    create_movie(Play1, debug=False, hq=True, output_file='play1.mp4')


if __name__ == '__main__':
    render_scene()
