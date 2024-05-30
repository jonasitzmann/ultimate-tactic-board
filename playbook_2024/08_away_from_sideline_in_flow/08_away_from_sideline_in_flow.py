from manim import *
from manim_animations import create_movie
from scenes import UltimateScene
import numpy as np


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        f.load_state(s[4])
        disc_arrow = f.get_disc_arrow(s[5], s[6])
        disc_text = Tex(r"Centering im\\Flow möglich", font_size=20).next_to(disc_arrow, UP, buff=0)
        disc_arrow_past = f.get_disc_arrow(s[1], s[2])[0]
        disc_arrow_past.put_start_and_end_on(disc_arrow_past.get_start(), f.get_player("o1").get_edge_center(LEFT))
        self.add(disc_arrow_past)
        self.add(f.get_arrows(s[1], s[4], players="o2"))
        self.add(disc_arrow, disc_text)
        d_arrow = f.get_arrows(s[4], s[5], players="d1", always=True)[0]
        arrow_direction = d_arrow.get_end() - d_arrow.get_start()
        arrow_direction /= np.linalg.norm(arrow_direction)
        arrow_end = d_arrow.get_end() + f.cs.scale_() * 2 * arrow_direction
        d_arrow.put_start_and_end_on(d_arrow.get_start(), arrow_end)
        self.add(Tex("Defense versucht, Power Position zu stoppen", font_size=20).next_to(d_arrow, DR, buff=0))
        self.add(d_arrow)


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        self.write("An der Sideline im Flow wieder in die Mitte spielen", animate=False)
        self.write_small("Ganzer Durchlauf", animate=False)
        self.wait()
        f.transition(s[1], run_time=0.7, disc_delay=0)
        f.transition(s[2], run_time=1, disc_delay=0, linear_end=True)
        f.transition(s[3], run_time=0.1)
        with f.field_of_view("o1", fade=False):
            f.transition(s[4], run_time=1, linear_start=True, floating=False)
            f.transition(s[5], run_time=0.7, fake="o2")
            f.transition(s[6], run_time=1.5, disc_delay=0)
        self.wait(2)

        f.transition(s[0], run_time=0.5)
        self.write_small(r"Nach Überläufer nach vorne schauen ist zwar erlaubt \dots")
        self.wait()
        f.transition(s[1], run_time=0.7, disc_delay=0)
        f.transition(s[2], run_time=1, disc_delay=0, linear_end=True)
        f.transition(s[3], run_time=0.1)
        with f.field_of_view("o1", fade=True):
            self.wait()
            self.write_small(
                r"""
            \dots aber falls dort niemand frei ist, wollen wir von der Sideline weg spielen,\\
            bevor die Situation statisch wird.
            """
            )
            self.wait()
            f.transition(s[4], run_time=1, linear_start=True, floating=False)

        with f.arrows(s[5], s[6], players="", disc=True, fade=True):
            self.wait()
            self.write_small(
                r"""
            Der Around Break ist leicht, wenn die Power Position stoppen möchte.\\
            Evtl. zusätzlich per Fake den Marker bewegen.
            """
            )
            self.wait(1)
            f.transition(s[5], run_time=0.7, fake="o2")
            f.transition(s[6], run_time=1.5, disc_delay=0)
        self.wait(2)


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
