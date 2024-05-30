from manim_animations import create_movie
from scenes import UltimateScene
from scenes.ultimate_scene import multi_context


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        player_labels = dict(o1="A", o2="B", o3="C")
        [f.get_player(name).set_label_text(label) for name, label in player_labels.items()]
        f.load_state(s[0])
        self.write("Sidelinebefreiung, Feldverteidiger*in overcommitted", animate=False)
        self.write_small("ganzes play")
        self.wait()
        f.transition(s[1], run_time=0.5)  # activation
        f.fake("o2")
        f.transition(s[2], run_time=0.7)  # prepare upline
        f.transition(s[3], run_time=0.7)  # upline
        f.fake("o2", run_time=0.1)
        f.transition(s[4], run_time=0.5, d_delay=0)  # prepare dump
        f.transition(s[5], run_time=1.2, disc_delay=0)  # dump
        f.transition(s[6], run_time=0.7)  # over-committment
        f.fake("o2")
        f.transition(s[7], run_time=1, disc_delay=0)  # counter
        self.wait(1)

        f.transition(s[5], run_time=0.2, d_delay=0)
        self.write_small(
            "Verteidiger*in im Feld versucht, breakside continuation zu verhindern\nsteht dadurch weit auf der breakside"
        )
        with f.highlight("d2"):
            self.wait(3)
            f.transition(s[6], run_time=0.7, d_delay=0)
            self.wait()
            self.write_small("counter: Cut auf die offene Seite")
            f.fake("o2")
            with f.arrows(next_state=s[7], players="o2"):
                self.wait(2)
                f.transition(s[7], run_time=1, disc_delay=0, d_delay=0.5)
                self.wait()
            self.wait()


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
