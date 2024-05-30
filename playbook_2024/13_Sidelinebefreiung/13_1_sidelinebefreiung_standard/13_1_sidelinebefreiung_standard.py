from manim_animations import create_movie
from scenes import UltimateScene
from scenes.ultimate_scene import multi_context


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        f.load_state(s[3])
        for i in range(2, 5):
            self.add(f.get_arrows(s[i], s[i + 1], players="o3", buffer=0))
        self.add(f.get_arrows(s[2], s[3], players="o2", buffer=0))
        self.add(f.get_arrows(s[3], s[7], players="o2", buffer=0))
        self.add(f.get_disc_arrow(s[4], s[5], buffer=0))
        self.add(f.get_disc_arrow(s[5], s[7], buffer=0))


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        player_labels = dict(o1="A", o2="B", o3="C")
        [f.get_player(name).set_label_text(label) for name, label in player_labels.items()]
        f.load_state(s[0])
        self.write("Sidelinebefreiung", animate=False)
        self.write_small("ganzes play")
        self.wait()
        f.transition(s[1], run_time=0.5)
        f.fake("o2")
        f.transition(s[2], run_time=0.7)
        f.transition(s[3], run_time=0.7, linear_end=True)
        f.fake("o2", run_time=0.1)
        f.transition(s[4], run_time=0.5, d_delay=0, linear_start=True, linear_end=True)
        f.transition(s[5], run_time=0.7, linear_start=True, linear_end=True, d_delay=0)
        f.transition(s[6], run_time=0.7, linear_start=True, d_delay=0)
        self.wait(1)

        f.transition(s[0], run_time=0.2)
        self.wait(1)
        self.write_small("B aktivieren")
        with f.field_of_view("o1"):
            self.wait(0.5)
            f.transition(s[1], run_time=0.5)
            self.wait(1)
            self.write_small("Nicht *sofort* rauslaufen, around break ermöglichen.")
            self.wait(1)
            f.fake("o2")
        self.write_small("B cleart raus per Überläufer. C repositioniert sich")
        self.wait(1)
        f.transition(s[2], run_time=0.7)
        f.transition(s[3], run_time=0.7)
        self.write_small("C bereitet Dump Cut vor!")
        self.wait(0.5)
        with multi_context(
            f.highlight("o3", fade=False),
            f.arrows(players="o3", next_state=s[4]),
            f.arrows(players="o3", state=s[4], next_state=s[5], fade=True),
        ):
            self.wait(2)
        self.write_small("B bereitet continuation vor")
        self.wait(0.5)
        with multi_context(
            f.highlight("o2", fade=False),
            f.arrows(players="o2", next_state=s[4]),
            f.arrows(players="o2", state=s[4], next_state=s[5], fade=True),
        ):
            self.wait(2)
        self.write_small("")
        f.transition(s[4], run_time=0.5, d_delay=0, linear_start=True)
        f.transition(s[5], run_time=0.7, linear_start=False, linear_end=True, d_delay=0)
        f.transition(s[6], run_time=0.7, linear_start=True, d_delay=0)
        self.wait(2)


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
