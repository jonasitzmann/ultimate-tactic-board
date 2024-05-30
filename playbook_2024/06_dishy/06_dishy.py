from manim_animations import create_movie
from scenes import UltimateScene


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        f.load_state(s[2])
        self.add(f.get_player("o4").get_field_of_view(field=f))
        self.add(f.get_arrows(s[2], s[3], players="o2"))
        self.add(f.get_disc_arrow(s[2], s[3]))


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        self.write("Nach kurzem Angebot: Ableger statt Pass nach vorne suchen", animate=False)
        self.write_small("Ganzes Play", animate=False)
        self.wait()
        f.transition(s[1], run_time=1)
        f.transition(s[2], run_time=1)
        f.transition(s[3], run_time=1, linear_end=True, disc_delay=0.2, o_delay=0.1, d_delay=0.2)
        f.transition(s[4], run_time=1, linear_start=True)
        self.wait()


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
