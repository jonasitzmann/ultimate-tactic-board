from manim_animations import create_movie, Field
from state import State
from manim import *
from flowchart import Decision, Action, FixArrow, focus, Group


class FlowchartNoDisc(Scene):
    def add(self, *mobjects):
        super().add(*mobjects)
        if len(mobjects) == 1:
            mobjects = mobjects[0]
        return mobjects

    def construct(self):
        s_buff = 0.38
        lw = 0.02
        stroke_width = DEFAULT_STROKE_WIDTH
        l_buff = LARGE_BUFF
        yes, no = [dict(color=c) for c in [GREEN, RED]]
        f = self.add(Field(self, State().setup_hex()).landscape_to_portrait(False))
        header = self.add(Tex("Verhalten abseits der Scheibe").to_edge(UP, s_buff).to_edge(LEFT))
        g_in_hand = Group()
        disc_in_hand = g_in_hand.add(self.add(Decision(r'Scheibe\\in der Hand?').next_to(header, DOWN, buff=s_buff)))
        in_hand_no = g_in_hand.add(FixArrow(ORIGIN, l_buff*LEFT, **no).shift(disc_in_hand.get_edge_center(LEFT)))
        self.add(g_in_hand)


        g_pos = Group()
        hex_pos = g_pos.add(Decision('An Hex Position?').next_to(disc_in_hand, DOWN, buff=s_buff))
        reposition = g_pos.add(Action(r'Repositionieren$^\ast$\\\scriptsize{Das ist ein Cut!}').next_to(hex_pos, RIGHT, buff=l_buff))
        hex_pos_no = g_pos.add(FixArrow(hex_pos.get_edge_center(RIGHT), reposition.get_edge_center(LEFT), **no))
        p_top_right = np.array([reposition.get_edge_center(RIGHT)[0] + s_buff, disc_in_hand.get_center()[1], 0])
        to_pos = g_pos.add(FixArrow(disc_in_hand.get_edge_center(DOWN), hex_pos.get_edge_center(UP), **yes))
        l_right = g_pos.add(Line(ORIGIN, s_buff*RIGHT).shift(reposition.get_edge_center(RIGHT)))
        l_up = g_pos.add(Line(l_right.get_end() + lw*DOWN, p_top_right))
        arr = g_pos.add(FixArrow(p_top_right + lw*RIGHT, disc_in_hand.get_edge_center(RIGHT)))
        self.add(g_pos)

        g_free = Group()
        free = g_free.add(Decision('Bist du frei?').next_to(hex_pos, DOWN, buff=s_buff))
        communicate = g_free.add(Action(r'Kommuniziere\\\scriptsize{Z.B. Arm heben}').next_to(free, RIGHT, buff=l_buff))
        free_yes = g_free.add(FixArrow(free.get_edge_center(RIGHT), communicate.get_edge_center(LEFT), **yes))
        to_free = g_free.add(FixArrow(hex_pos.get_edge_center(DOWN), free.get_edge_center(UP), **yes))
        l_right = g_pos.add(Line(ORIGIN, s_buff*RIGHT).shift(communicate.get_edge_center(RIGHT)))
        l_up = g_pos.add(Line(l_right.get_end() + lw*DOWN, p_top_right))
        self.add(g_free)

        g_looking = Group()
        thrower_looking = g_looking.add(Decision(r'Schaut Werfer\\dich an?').next_to(free, DOWN, buff=s_buff))
        generate_option = g_looking.add(Action(r'Mach was$^\ast$\\\scriptsize{Ggf. nur Platz machen}').next_to(thrower_looking, RIGHT, buff=l_buff))
        thrower_looking_yes = g_looking.add(FixArrow(thrower_looking.get_edge_center(RIGHT), generate_option.get_edge_center(LEFT), **yes))
        to_looking = g_free.add(FixArrow(free.get_edge_center(DOWN), thrower_looking.get_edge_center(UP), **no))
        l_right = g_pos.add(Line(ORIGIN, s_buff*RIGHT).shift(generate_option.get_edge_center(RIGHT)))
        l_up = g_pos.add(Line(l_right.get_end() + lw*DOWN, p_top_right))
        self.add(g_looking)

        g_space = Group()
        create_space = g_space.add(Decision(r'Platz machen\\sinnvoll?').next_to(thrower_looking, DOWN, buff=s_buff))
        chill = g_space.add(Action(r'Chill$^\ast$\\\scriptsize{Sammle Infos}').next_to(create_space, RIGHT, buff=l_buff))
        chill.shift(DOWN * (chill.get_edge_center(UP)[1] - create_space.get_edge_center(DOWN)[1]))
        space_yes = g_space.add(Group())
        p = np.array([generate_option.get_center()[0], create_space.get_center()[1], 0])
        l2 = space_yes.add(FixArrow(p, generate_option.get_edge_center(DOWN), **yes))
        l1 = space_yes.add(Line(create_space.get_edge_center(RIGHT), p + lw * RIGHT, **yes, stroke_width=stroke_width))
        space_no = g_space.add(Group())
        p = np.array([create_space.get_center()[0], chill.get_center()[1], 0])
        l1 = space_no.add(Line(create_space.get_edge_center(DOWN), p, **no, stroke_width=stroke_width))
        l2 = space_no.add(FixArrow(p + lw*LEFT, chill.get_edge_center(LEFT), **no))
        to_space = g_free.add(FixArrow(thrower_looking.get_edge_center(DOWN), create_space.get_edge_center(UP), **no))
        l_right = g_pos.add(Line(ORIGIN, s_buff*RIGHT).shift(chill.get_edge_center(RIGHT)))
        l_up = g_pos.add(Line(l_right.get_end() + lw*DOWN, p_top_right))
        self.add(g_space)


if __name__ == '__main__':
    create_movie(FlowchartNoDisc, debug=False, hq=True, output_file='flowchart_no_disc.mp4')
