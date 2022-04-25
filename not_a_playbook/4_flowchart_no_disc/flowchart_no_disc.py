from manim_animations import create_movie, Field
from manim import *
from flowchart import Decision, Action, FixArrow, focus, Group, get_focus_mobject
import flowchart
import cfg
from manim_presentation import Slide


class FlowchartNoDisc(Slide):
    def add(self, *mobjects):
        super().add(*mobjects)
        if len(mobjects) == 1:
            mobjects = mobjects[0]
        return mobjects

    def pplay(self, *args, **kwargs):
        self.play(*args, **kwargs)
        self.pause()

    def construct(self):
        cfg.player_scale = 1.3
        s_buff = 0.38
        lw = 0.02
        stroke_width = DEFAULT_STROKE_WIDTH
        l_buff = LARGE_BUFF
        yes, no = [dict(color=c) for c in [GREEN, RED]]
        self.wait(0.1)
        self.pause()
        header = self.add(Tex("Verhalten abseits der Scheibe").to_edge(UP, s_buff).to_edge(LEFT))
        self.play(Write(header))
        self.pause()
        g_in_hand = Group()
        disc_in_hand = g_in_hand.add(Decision(r'Scheibe\\in der Hand?').next_to(header, DOWN, buff=s_buff))
        in_hand_no = g_in_hand.add(FixArrow(ORIGIN, l_buff*LEFT, **yes).shift(disc_in_hand.get_edge_center(LEFT)))

        # are you in correct hex position?
        g_pos = Group()
        hex_pos = g_pos.add(Decision('An Hex Position?').next_to(disc_in_hand, DOWN, buff=s_buff))
        reposition = g_pos.add(Action(r'Repositionieren$^\ast$\\\scriptsize{Das ist ein Cut!}').next_to(hex_pos, RIGHT, buff=l_buff))
        hex_pos_no = g_pos.add(FixArrow(hex_pos.get_edge_center(RIGHT), reposition.get_edge_center(LEFT), **no))
        p_top_right = np.array([reposition.get_edge_center(RIGHT)[0] + s_buff, disc_in_hand.get_center()[1], 0])
        to_pos = g_pos.add(FixArrow(disc_in_hand.get_edge_center(DOWN), hex_pos.get_edge_center(UP), **no))
        l_right = g_pos.add(Line(ORIGIN, s_buff*RIGHT).shift(reposition.get_edge_center(RIGHT)))
        l_up = g_pos.add(Line(l_right.get_end() + lw*DOWN, p_top_right))
        arr = g_pos.add(FixArrow(p_top_right + lw*RIGHT, disc_in_hand.get_edge_center(RIGHT)))
        self.add(g_pos, g_in_hand)
        f: Field = self.add(Field(self, 'states/reposition_1.yaml').landscape_to_portrait(False))
        with f.highlight('o4', fade=False):
            self.pplay(Write(g_pos), Write(g_in_hand), Write(f), run_time=1)
            flowchart.focus_mobject = get_focus_mobject(disc_in_hand)
            self.add(flowchart.focus_mobject)
            [self.pplay(focus(mob)) for mob in [hex_pos, reposition]]
            f.transition('states/reposition_2.yaml')
            self.pplay(focus(disc_in_hand))

        # are you open?
        g_free = Group()
        free = g_free.add(Decision('Bist du frei?').next_to(hex_pos, DOWN, buff=s_buff))
        communicate = g_free.add(Action(r'Kommuniziere\\\scriptsize{Z.B. Arm heben}').next_to(free, RIGHT, buff=l_buff))
        free_yes = g_free.add(FixArrow(free.get_edge_center(RIGHT), communicate.get_edge_center(LEFT), **yes))
        to_free = g_free.add(FixArrow(hex_pos.get_edge_center(DOWN), free.get_edge_center(UP), **yes))
        l_right = g_free.add(Line(ORIGIN, s_buff * RIGHT).shift(communicate.get_edge_center(RIGHT)))
        l_up = g_free.add(Line(l_right.get_end() + lw * DOWN, p_top_right))
        with f.highlight('o6', fade=False):
            f.transition('states/open_1.yaml', run_time=0.5)
            self.pause()
            f.transition('states/open_2.yaml')
            self.pplay(focus(hex_pos))
            self.play(Write(g_free))
            [self.pplay(focus(mob)) for mob in [free, communicate]]
            [self.play(Indicate(f.get_player('o6')), run_time=0.5) for _ in range(3)]
            f.transition('states/open_3.yaml')
            self.pause()
            self.pplay(focus(disc_in_hand))
        self.add(g_free)

        g_looking = Group()
        thrower_looking = g_looking.add(Decision(r'Schaut Werfer\\dich an?').next_to(free, DOWN, buff=s_buff))
        generate_option = g_looking.add(Action(r'Mach was$^\ast$\\\scriptsize{Ggf. nur Platz machen}').next_to(thrower_looking, RIGHT, buff=l_buff))
        thrower_looking_yes = g_looking.add(FixArrow(thrower_looking.get_edge_center(RIGHT), generate_option.get_edge_center(LEFT), **yes))
        to_looking = g_looking.add(FixArrow(free.get_edge_center(DOWN), thrower_looking.get_edge_center(UP), **no))
        l_right = g_looking.add(Line(ORIGIN, s_buff*RIGHT).shift(generate_option.get_edge_center(RIGHT)))
        l_up = g_looking.add(Line(l_right.get_end() + lw*DOWN, p_top_right))

        for i, alternative in enumerate([2, 3]):
            f.transition('states/looking_1.yaml', run_time=0.5)
            with f.highlight('o2', fade=False):
                self.wait(0.1)
                self.pause()
                [self.pplay(focus(mob)) for mob in [hex_pos, free]]
                if i == 0:
                    self.play(Write(g_looking))
                self.pplay(focus(thrower_looking))
                with f.field_of_view('o1'):
                    self.pplay(focus(generate_option))
                    f.transition(f'states/looking_{alternative}.yaml')
                    self.pause()
                    self.pplay(focus(disc_in_hand))
        self.add(g_looking)

        g_space = Group()
        create_space = g_space.add(Decision(r'Platz machen\\sinnvoll?').next_to(thrower_looking, DOWN, buff=s_buff))
        space_yes = g_space.add(Group())
        p = np.array([generate_option.get_center()[0], create_space.get_center()[1], 0])
        l2 = space_yes.add(FixArrow(p, generate_option.get_edge_center(DOWN), **yes))
        l1 = space_yes.add(Line(create_space.get_edge_center(RIGHT), p + lw * RIGHT, **yes, stroke_width=stroke_width))
        to_space = g_space.add(FixArrow(thrower_looking.get_edge_center(DOWN), create_space.get_edge_center(UP), **no))

        g_chill = Group()
        chill = g_chill.add(Action(r'Chill$^\ast$\\\scriptsize{Sammle Infos}').next_to(create_space, RIGHT, buff=l_buff))
        chill.shift(DOWN * (chill.get_edge_center(UP)[1] - create_space.get_edge_center(DOWN)[1]))
        p = np.array([create_space.get_center()[0], chill.get_center()[1], 0])
        space_no = g_chill.add(Group())
        l1 = space_no.add(Line(create_space.get_edge_center(DOWN), p, **no, stroke_width=stroke_width))
        l2 = space_no.add(FixArrow(p + lw*LEFT, chill.get_edge_center(LEFT), **no))
        l_right = g_chill.add(Line(ORIGIN, s_buff*RIGHT).shift(chill.get_edge_center(RIGHT)))
        l_up = g_chill.add(Line(l_right.get_end() + lw*DOWN, p_top_right))

        with f.highlight('o5', fade=False):
            f.transition('states/looking_1.yaml', run_time=0.5)
            self.pause()
            [self.pplay(focus(mob)) for mob in [hex_pos, free, thrower_looking]]
            with f.field_of_view('o1'):
                self.play(Write(g_space))
                [self.pplay(focus(mob)) for mob in [create_space, generate_option]]
                f.transition('states/looking_2.yaml')
                self.pause()
                self.pplay(focus(disc_in_hand))
        self.add(g_space)

        with f.highlight('o3', fade=False):
            f.transition('states/looking_1.yaml', run_time=0.5)
            self.pause()
            [self.pplay(focus(mob)) for mob in [hex_pos, free, thrower_looking, create_space]]
            self.play(Write(g_chill))
            self.pplay(focus(chill))
            with f.field_of_view('o3'):
                self.pause()
                f.transition('states/looking_2.yaml')
                self.pause()
            self.pplay(focus(disc_in_hand))
            [self.pplay(focus(mob)) for mob in [hex_pos, reposition]]
        self.add(g_chill)
        self.wait()


if __name__ == '__main__':
    create_movie(FlowchartNoDisc, debug=False, hq=True, output_file='flowchart_with_disc.mp4')
    bin_dir = '/home/jonas/.conda/envs/tactics_board/bin'
    os.system(f'{bin_dir}/manim-presentation FlowchartNoDisc --fullscreen')
