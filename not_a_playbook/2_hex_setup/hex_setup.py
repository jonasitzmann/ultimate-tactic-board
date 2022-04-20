from manim import *
from manim_animations import Field, Player, create_movie, myblue
from state import State
import cfg


class HexSetup(Scene):
    def construct(self):
        tex = Tex('Aufstellung 4v4 bis 7v7').to_edge(UP)
        self.play(Write(tex))
        self.wait(1)
        for x, y in [(cfg.field_width_m / 2, 20), (cfg.field_width_m / 4, 40), (0, 60)]:
            state = State()
            state.setup_hex(pos=np.array([x, y]))
            f = Field(self, State(disc=state.disc)).to_edge(DOWN)
            connected_pairs = [
                (1, 4), (1, 2), (4, 2), (1, 5), (5, 2),
                (4, 6),
                (6, 3), (6, 2),
                (3, 7), (3, 2),
                (7, 5), (7, 2),
            ]
            lines = [Line(*[f.cs.c2p(*state.find_player(f'o{x}').pos) for x in pair], color=WHITE, stroke_opacity=0.3) for pair in connected_pairs]
            self.add(f)
            self.play(Transform(tex, Tex('Ermittlung der Spielrichtung').to_edge(UP)))
            scale = f.cs.scale_()
            f.set_player(state.find_player('o1'))
            f.add(f.get_player('o1').get_field_of_view(f))
            vector_field = f.get_play_direction_arrows()

            self.play(FadeIn(vector_field))
            self.wait(2)
            arrow = Arrow(stroke_width=20 * f.cs.scale_(), max_tip_length_to_length_ratio=0.1, color=YELLOW)
            arrow.put_start_and_end_on(f.cs.c2p(*state.find_player('o1').pos), f.cs.c2p(*state.find_player('o2').pos))
            arrow.account_for_buff(0.5 * f.cs.scale_())
            arrow.remove(arrow.tip)
            arrow.add_tip()
            # play_direction_text = Tex('Spielrichtung', color=YELLOW).rotate(arrow.get_angle()).scale(3*scale).move_to(arrow)
            self.play(Transform(tex, Tex('Hat in Spielrichtung ausgerichtet').to_edge(UP)))
            self.play(Create(arrow))
            self.wait()
            self.play(FadeOut(vector_field))
            f.set_player(state.find_player('o2'))
            hat_label = Tex(r'Hat').scale(3*scale).move_to(f.cs.c2p(*state.find_player('o2').pos)).shift(1.5*UP*scale)
            f.add(hat_label)
            self.wait()
            self.play(FadeOut(arrow))
            self.wait()
            front_players = VGroup(*[f.set_player(state.find_player(f'o{x}')) for x in [5, 4]])
            self.play(Transform(tex, Tex('Scheibennahe Spieler bilden Dreiecke mit Hat').to_edge(UP)))
            self.play(*[FadeIn(l) for l in lines[:5]])
            self.wait(2)
            back_players = VGroup(*[f.set_player(state.find_player(f'o{x}')) for x in [3, 6, 7]])
            self.play(Transform(tex, Tex('Scheibenferne Spieler (bei 5v5-7v7) verteilen sich dahinter').to_edge(UP)))
            self.play(*[FadeIn(l) for l in lines[5:]])
            self.wait(4)
            self.remove(f, *[l for l in lines])


def render_scene():
    create_movie(HexSetup, debug=False, hq=True, output_file='hex_setup.mp4')


if __name__ == '__main__':
    render_scene()
