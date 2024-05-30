from manim_animations import create_movie
from scenes import UltimateScene
from manim import *


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        f.load_state(s[0])
        self.add(f.get_arrows(s[0], s[1], players="o5 o6", buffer=0))
        self.add(f.get_arrows(s[1], s[2], players="o5 o6", buffer=0))
        self.add(Tex("Offene Seite", font_size=20).next_to(f.cs.c2p([[37, 30]]), UR))
        self.add(Tex("Geschlossene Seite", font_size=20).next_to(f.cs.c2p([[0, 30]]), DR))
        # angle, label = f.get_angle_measurement(f.get_player("o1"), f.get_player("o2"))
        # angle.set_opacity(0.3)
        # self.add(label, angle)


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        self.write("Aufstellung nach Brick. Cutting-initiiert", animate=False)
        self.write_small("Ganzer Durchlauf", animate=False)
        self.wait()
        f.transition(s[0], run_time=1)
        f.transition(s[1], run_time=1)
        f.transition(s[2], run_time=1)
        self.wait()


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
