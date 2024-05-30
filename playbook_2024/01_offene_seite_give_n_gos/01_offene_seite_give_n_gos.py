from manim_animations import create_movie
from scenes import UltimateScene
from manim import *


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        self.write(r"Offene Seite: Give'n Go (A $\rightarrow$ B $\rightarrow$ A)", animate=False)
        f.add(Tex("geschlossene Seite", font_size=20).move_to(f.cs.c2p([[12, 22]]), UL))
        f.add(Tex("offene Seite", font_size=20).move_to(f.cs.c2p([[33, 22]]), DL))
        player_labels = dict(o1="A", o2="B", o3="C")
        [f.get_player(name).set_label_text(label) for name, label in player_labels.items()]
        f.load_state(s[1])
        self.add(f.get_arrows(s[0], s[1], players="o2"))
        self.add(f.get_arrows(s[1], s[2], players="o1"))
        self.add(f.get_disc_arrow(s[0], s[1]))
        self.add(f.get_disc_arrow(s[1], s[2]))


class Play1(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        player_labels = dict(o1="A", o2="B", o3="C")
        [f.get_player(name).set_label_text(label) for name, label in player_labels.items()]
        self.write(r"Offene Seite: Give'n Go (A $\rightarrow$ B $\rightarrow$ A)", animate=False)
        self.write_small("Durchlauf")
        self.wait()
        f.transition(s[1], run_time=1, linear_end=True, disc_delay=0, o_delay=0.1, floating=False)
        f.transition(s[2], run_time=1, linear_start=True)
        self.wait()
        f.transition(s[0], run_time=0.3)
        self.write_small("Pass auf die offene Seite wird gespielt")
        self.wait()
        f.transition(s[1], run_time=1)
        self.write_small("A ist auf der offenen Seite automatisch frei.")
        with f.arrows(s[1], s[2], players="o1"):
            self.wait()
            self.write_small("Wenn Platz auf der offenen Seite: Immer hinterher laufen!")
            self.wait(2)
            f.transition(s[2], run_time=1)
        self.wait(2)


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Play1, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
