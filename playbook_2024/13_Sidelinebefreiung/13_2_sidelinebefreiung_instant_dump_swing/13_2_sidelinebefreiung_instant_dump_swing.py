from manim_animations import create_movie
from scenes import UltimateScene
from manim import *


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        player_labels = dict(o1="A", o2="B", o3="C")
        [f.get_player(name).set_label_text(label) for name, label in player_labels.items()]
        self.write("Instant Dump Swing", animate=False)

        self.write_small("ganzes play")
        self.wait(1)
        f.load_state(s[1])
        f.transition(s[2], run_time=0.7, linear_end=True)
        f.transition(s[3], run_time=0.7, linear_start=True, linear_end=True)
        f.transition(s[5], run_time=1, linear_start=True)
        self.wait(1)

        f.transition(s[0], run_time=0.2)
        self.wait()
        self.write_small("Dump aktivieren")
        with f.field_of_view("o1"):
            self.wait(0.5)
            f.transition(s[1], run_time=0.5)
            self.wait(0.5)
        self.write_small("breakside cut vorbereiten")  # (Auf scheibe zu, falls Defense nicht committet -> Überläufer)')
        with f.arrows(players="o2", next_state=s[2]), f.arrows(players="o2", state=s[2], next_state=s[3]):
            self.wait(2)
            f.transition(s[2], run_time=0.5, linear_end=True)
            f.transition(s[3], run_time=0.7, linear_start=True)

        self.write_small("nach break Pass repositionieren")  #  (45° hinter der Scheibe auf der offenen Seite)')
        with f.arrows(players="o1", next_state=s[4]):
            self.wait(0.5)
            f.transition(s[4], run_time=1)

        # self.write_small('breakside cut vorbereiten')
        self.write_small("")
        with f.arrows(players="o3", state=s[2], next_state=s[3]), f.arrows(players="o3", state=s[4], next_state=s[5]):
            self.wait(1)
            f.transition(s[5], run_time=1)
        self.wait(2)


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
