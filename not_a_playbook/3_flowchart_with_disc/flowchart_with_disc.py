import sys
sys.path.append('../../')
from manim_animations import create_movie, Field
from state import State
from manim import *
from flowchart import Decision, Action, FixArrow, focus, Group, get_focus_mobject
import flowchart
from manim_presentation import Slide


class FlowchartWithDisc(Slide):
    @staticmethod
    def get_state(play_number, state_number):
        return State.load(f'{play_number}/{state_number}.yaml')

    def add(self, *mobjects):
        super().add(*mobjects)
        if len(mobjects) == 1:
            mobjects = mobjects[0]
        return mobjects

    def pplay(self, *args, **kwargs):
        self.play(*args, **kwargs)
        self.pause()

    def construct(self):
        stroke_width = DEFAULT_STROKE_WIDTH
        yes, no, maybe = [dict(color=c) for c in [GREEN, RED, YELLOW]]
        buff = MED_LARGE_BUFF
        orientation = DOWN
        lw = 0.02

        f = Field(self, self.get_state(1, 1)).landscape_to_portrait(False)
        self.wait(0.1)
        self.pause()
        header = self.add(Tex("Einfache Anspieloptionen finden").to_edge(UP, MED_SMALL_BUFF).to_edge(LEFT))
        self.play(Write(header))
        self.pause()

        g_basic = Group()
        disc_in_hand = Decision(r'Scheibe\\in der Hand?').next_to(header, DOWN, buff=MED_LARGE_BUFF).shift(RIGHT)
        in_front = Decision(r'Jemand\\vor dir frei?').next_to(disc_in_hand, orientation, buff=buff)
        decisions = [disc_in_hand, in_front]
        g_basic.add(*decisions)
        g_basic.add(FixArrow(disc_in_hand.get_corner(orientation), in_front.get_corner(orientation * -1), **yes))
        throw_pass = Action(r'Nimm den\\offenen Pass').next_to(in_front, LEFT, LARGE_BUFF)
        return_pass = Action(r'Biete dich für\\Rückpass an').next_to(disc_in_hand, LEFT, buff=LARGE_BUFF)
        take_pass_mobjects = [throw_pass, return_pass]
        take_pass_mobjects.append(FixArrow(throw_pass.get_edge_center(UP), return_pass.get_edge_center(DOWN)))
        take_pass_mobjects.append(FixArrow(return_pass.get_edge_center(RIGHT), disc_in_hand.get_edge_center(LEFT)))
        take_pass_mobjects.append(FixArrow(in_front.get_edge_center(LEFT), throw_pass.get_edge_center(RIGHT), **yes))
        take_pass_x = throw_pass.get_center()[0]
        take_pass_dx = take_pass_x - disc_in_hand.get_corner(LEFT)[0]
        g_basic.add(*take_pass_mobjects)
        dots = Action(r'\dots').next_to(disc_in_hand, RIGHT, buff=LARGE_BUFF)
        g_basic.add(dots)
        g_basic.add(FixArrow(disc_in_hand.get_edge_center(RIGHT), dots.get_edge_center(LEFT), **no))

        f.transition(self.get_state(1, 2), run_time=1)
        f.fake('o2')
        f.transition(self.get_state(1, 3), run_time=2, d_delay=0.3)
        f.transition(self.get_state(1, 4), run_time=2, disc_delay=0, d_delay=0.3, linear_end=True)
        f.transition(self.get_state(1, 5), run_time=2, disc_delay=0, linear_start=True)
        f.transition(self.get_state(1, 6), run_time=2)
        f.transition(self.get_state(1, 7), run_time=2)
        f.transition(self.get_state(1, 8), run_time=2)
        self.pause()

        f.transition(self.get_state(1, 1), run_time=0.5)
        with f.highlight('o2', fade=False):
            flowchart.focus_mobject = get_focus_mobject(disc_in_hand)
            self.add(flowchart.focus_mobject)
            self.pplay(Write(g_basic), Write(flowchart.focus_mobject))
            f.transition(self.get_state(1, 2), run_time=1)
            f.fake('o2')
            f.transition(self.get_state(1, 3), run_time=2, d_delay=0.3)
            self.pause()
            self.pplay(focus(in_front))
            with f.field_of_view('o2'):
                self.pause()
                [self.pplay(focus(mob)) for mob in [throw_pass, return_pass, disc_in_hand]]
            f.transition(self.get_state(1, 4), run_time=2, disc_delay=0, d_delay=0.3, linear_end=True)
            f.transition(self.get_state(1, 5), run_time=2, disc_delay=0, linear_start=True)
            self.pplay(focus(disc_in_hand))
        self.add(g_basic)


        g_fake = Group()
        fake = Action('Fake').next_to(in_front, RIGHT, buff=LARGE_BUFF)
        g_fake.add(fake)
        fake_x = fake.get_center()[0]
        fake_dx = fake_x - disc_in_hand.get_corner(RIGHT)[0]
        g_fake.add(FixArrow(in_front.get_edge_center(RIGHT), fake.get_edge_center(LEFT), **maybe))
        dx = in_front.get_center()[0] - fake_x
        d_y = (disc_in_hand.get_edge_center(DOWN)[1] - fake.get_edge_center(UP)[1]) * 0.7
        p1_fake = fake.get_edge_center(UP) + d_y * UP
        p2 = p1_fake + dx * RIGHT
        g_fake.add(Line(fake.get_edge_center(UP), p1_fake))
        g_fake.add(Line(p1_fake, p2, z_index=-1))
        f.transition(self.get_state(1, 1), run_time=0.5)
        with f.highlight('o1', fade=False):
            self.wait(0.1)
            self.pause()
            self.pplay(focus(in_front))
            with f.field_of_view('o1'):
                self.pause()
                self.play(Write(g_fake), run_time=0.5)
                self.wait()
                self.pplay(focus(fake))
                f.transition(self.get_state(1, 2), run_time=1)
                f.fake('o2')
                f.transition(self.get_state(1, 3), run_time=2, d_delay=0.3)
                self.pause()
                [self.pplay(focus(mob)) for mob in [in_front, throw_pass, return_pass, disc_in_hand]]
        self.add(g_fake)

        prev_thrower = Decision(r'Letzter\\Werfer frei?').next_to(decisions[-1], orientation, buff=buff)
        g_prev = Group(prev_thrower)
        g_prev.add(FixArrow(decisions[-1].get_corner(orientation), prev_thrower.get_corner(UP), **no))
        decisions.append(prev_thrower)
        p1 = prev_thrower.get_corner(RIGHT)
        p2 = p1 + fake_dx * RIGHT
        fake_line = Line(p1, p2, stroke_width=stroke_width, **maybe)
        g_prev.add(fake_line)
        g_prev.add(FixArrow(fake_line.get_end() + lw*DOWN, fake.get_edge_center(DOWN), **maybe))
        l1 = Line(prev_thrower.get_corner(LEFT), prev_thrower.get_corner(LEFT) + take_pass_dx * RIGHT, stroke_width=stroke_width, **yes)
        l2 = FixArrow(l1.get_end() + lw*DOWN, throw_pass.get_edge_center(DOWN), **yes)
        g_prev.add(l1, l2)
        f.transition(self.get_state(1, 1), run_time=0.5)
        with f.highlight('o6', fade=False):
            self.wait(0.1)
            self.pause()
            f.transition(self.get_state(1, 2), run_time=1)
            f.transition(self.get_state(1, 3), run_time=2, d_delay=0.3)
            f.transition(self.get_state(1, 4), run_time=2, disc_delay=0, d_delay=0.3, linear_end=True)
            self.pause()
            self.pplay(focus(in_front))
            self.play(Write(g_prev), run_time=0.5)
            self.pplay(focus(prev_thrower))
            [self.pplay(focus(mob)) for mob in [throw_pass, return_pass, disc_in_hand]]
            f.transition(self.get_state(1, 5), run_time=2, disc_delay=0, linear_start=True)
        self.add(g_prev)

        continuation = Decision(r'Continuation\\möglich?').next_to(decisions[-1], orientation, buff=buff)
        g_cont = Group(continuation)
        g_cont.add(FixArrow(decisions[-1].get_corner(orientation), continuation.get_corner(UP), **no))
        decisions.append(continuation)
        p1 = continuation.get_corner(RIGHT)
        p2 = p1 + fake_dx * RIGHT
        fake_line = Line(p1, p2, stroke_width=stroke_width, **maybe)
        g_cont.add(fake_line)
        g_cont.add(FixArrow(fake_line.get_end() + lw*DOWN, fake.get_edge_center(DOWN), **maybe))
        l1 = Line(continuation.get_corner(LEFT), continuation.get_corner(LEFT) + take_pass_dx * RIGHT, stroke_width=stroke_width, **yes)
        l2 = FixArrow(l1.get_end() + lw*DOWN, throw_pass.get_edge_center(DOWN), **yes)
        g_cont.add(l1, l2)

        face_center = Action(r'Blicke Richtung\\Spielfeldmitte').next_to(decisions[-1], orientation, buff=buff)
        g_center = VGroup(face_center)
        g_center.add(FixArrow(decisions[-1].get_corner(DOWN), face_center.get_corner(UP), **no))
        p1 = np.array([fake.get_edge_center(RIGHT)[0] + MED_SMALL_BUFF, face_center.get_center()[1], 0])
        g_center.add(Line(face_center.get_edge_center(RIGHT), p1))
        p2 = np.array([p1[0], p1_fake[1], 0])
        g_center.add(Line(p1 + lw * DOWN, p2 + lw * UP))
        g_center.add(Line(p2, p1_fake))
        f.load_state(self.get_state(1, 5))
        with f.highlight('o2', fade=False):
            self.wait(0.1)
            self.pause()
            [self.pplay(focus(mob)) for mob in [in_front, prev_thrower]]
            self.play(Write(g_cont), run_time=0.5)
            self.pplay(focus(continuation))
            self.play(Write(g_center), run_time=0.5)
            self.pplay(focus(face_center))
            with f.field_of_view('o2'):
                f.transition(self.get_state(1, 6), run_time=2)
                self.pause()
                [self.pplay(focus(mob)) for mob in [in_front, throw_pass, return_pass, disc_in_hand]]
                f.transition(self.get_state(1, 7), run_time=2)
            self.play(focus(disc_in_hand))

        with f.highlight('o4', fade=False):
            self.wait(0.1)
            [self.pplay(focus(mob)) for mob in [in_front, prev_thrower, continuation, throw_pass, return_pass, disc_in_hand]]
            f.transition(self.get_state(1, 8), run_time=2)
        self.pause()
        self.wait()



if __name__ == '__main__':
    # create_movie(FlowchartWithDisc, debug=False, hq=True, output_file='flowchart_with_disc.mp4')
    bin_dir = '/home/jonas/.conda/envs/tactics_board/bin'
    os.system(f'{bin_dir}/manim-presentation FlowchartWithDisc --fullscreen')
