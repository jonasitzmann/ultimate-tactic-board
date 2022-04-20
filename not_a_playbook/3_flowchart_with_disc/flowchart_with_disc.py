from manim_animations import create_movie, Field
from state import State
from manim import *
from flowchart import Decision, Action, FixArrow, focus


class FlowchartWithDisc(Scene):
    def construct(self):
        stroke_width = DEFAULT_STROKE_WIDTH
        group = VGroup()
        yes_color, no_color, maybe_color = GREEN, RED, YELLOW
        yes, no, maybe = [dict(color=c) for c in [yes_color, no_color, maybe_color]]
        f = Field(self, State().setup_hex()).landscape_to_portrait(False)
        buff = MED_LARGE_BUFF
        orientation = DOWN
        header = Tex("Einfache Anspieloptionen finden").to_edge(UP, MED_SMALL_BUFF).to_edge(LEFT)
        self.add(header)
        disc_in_hand = Decision(r'Scheibe\\in der Hand?').next_to(header, DOWN, buff=MED_LARGE_BUFF).shift(RIGHT)
        in_front = Decision(r'Jemand\\vor dir frei?').next_to(disc_in_hand, orientation, buff=buff)
        decisions = [disc_in_hand, in_front]
        group.add(*decisions)
        extensions = [r'Letzter\\Werfer frei?', r'Continuation\\möglich?']

        dots = Action(r'\dots').next_to(disc_in_hand, RIGHT, buff=LARGE_BUFF)
        group.add(dots)
        group.add(FixArrow(disc_in_hand.get_edge_center(RIGHT), dots.get_edge_center(LEFT), **no))
        group.add(FixArrow(disc_in_hand.get_corner(orientation), in_front.get_corner(orientation * -1), **yes))


        throw_pass = Action(r'Nimm den\\offenen Pass').next_to(in_front, LEFT, LARGE_BUFF)
        return_pass = Action(r'Biete dich für\\Rückpass an').next_to(disc_in_hand, LEFT, buff=LARGE_BUFF)
        take_pass_mobjects = [throw_pass, return_pass]
        take_pass_mobjects.append(FixArrow(throw_pass.get_edge_center(UP), return_pass.get_edge_center(DOWN)))
        take_pass_mobjects.append(FixArrow(return_pass.get_edge_center(RIGHT), disc_in_hand.get_edge_center(LEFT)))
        take_pass_mobjects.append(FixArrow(in_front.get_edge_center(LEFT), throw_pass.get_edge_center(RIGHT), **yes))
        take_pass_x = throw_pass.get_center()[0]
        take_pass_dx = take_pass_x - disc_in_hand.get_corner(LEFT)[0]
        group.add(*take_pass_mobjects)
        self.play(Write(group), run_time=0.5)
        self.wait(1)
        self.add(group)
        for mob in [*decisions, throw_pass, return_pass, return_pass, disc_in_hand]:
            self.play(focus(mob))
        self.wait()

        group = VGroup()
        fake = Action('Fake').next_to(in_front, RIGHT, buff=LARGE_BUFF)
        group.add(fake)
        fake_x = fake.get_center()[0]
        fake_dx = fake_x - disc_in_hand.get_corner(RIGHT)[0]
        group.add(FixArrow(in_front.get_edge_center(RIGHT), fake.get_edge_center(LEFT), **maybe))
        dx = in_front.get_center()[0] - fake_x
        d_y = (disc_in_hand.get_edge_center(DOWN)[1] - fake.get_edge_center(UP)[1]) * 0.7
        p1_fake = fake.get_edge_center(UP) + d_y * UP
        p2 = p1_fake + dx * RIGHT
        lw = 0.02
        group.add(Line(fake.get_edge_center(UP), p1_fake))
        group.add(Line(p1_fake, p2, z_index=-1))
        self.play(Write(group), run_time=0.5)
        self.wait(1)
        self.add(group)
        for mob in [*decisions, fake]:
            self.play(focus(mob))
        self.wait()
        for mob in [in_front, throw_pass, return_pass, return_pass, disc_in_hand]:
            self.play(focus(mob))
        self.wait()

        for extension in extensions:
            decision = Decision(extension).next_to(decisions[-1], orientation, buff=buff)
            group = VGroup(decision)
            group.add(FixArrow(decisions[-1].get_corner(orientation), decision.get_corner(UP), **no))
            decisions.append(decision)
            p1 = decision.get_corner(RIGHT)
            p2 = p1 + fake_dx * RIGHT
            fake_line = Line(p1, p2, stroke_width=stroke_width, **maybe)
            group.add(fake_line)
            group.add(FixArrow(fake_line.get_end() + lw*DOWN, fake.get_edge_center(DOWN), **maybe))
            l1 = Line(decision.get_corner(LEFT), decision.get_corner(LEFT) + take_pass_dx * RIGHT, stroke_width=stroke_width, **yes)
            l2 = FixArrow(l1.get_end() + lw*DOWN, throw_pass.get_edge_center(DOWN), **yes)
            group.add(l1, l2)
            self.play(Write(group), run_time=0.5)
            self.wait()
            for mob in [*decisions, throw_pass, return_pass, return_pass, disc_in_hand]:
                self.play(focus(mob))
            self.add(group)
            self.wait()

        face_center = Action(r'Blicke Richtung\\Spielfeldmitte').next_to(decisions[-1], orientation, buff=buff)
        group = VGroup(face_center)
        group.add(FixArrow(decisions[-1].get_corner(DOWN), face_center.get_corner(UP), **no))
        p1 = np.array([fake.get_edge_center(RIGHT)[0] + MED_SMALL_BUFF, face_center.get_center()[1], 0])
        group.add(Line(face_center.get_edge_center(RIGHT), p1))
        p2 = np.array([p1[0], p1_fake[1], 0])
        group.add(Line(p1 + lw * DOWN, p2 + lw * UP))
        group.add(Line(p2, p1_fake))
        self.play(Write(group), run_time=0.5)
        self.wait()
        for mob in [*decisions, face_center, face_center, in_front]:
            self.play(focus(mob))
        self.add(group)
        self.wait(2)


if __name__ == '__main__':
    create_movie(FlowchartWithDisc, debug=False, hq=True, output_file='play_name.mp4')
