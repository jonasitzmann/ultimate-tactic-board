from manim_animations import create_movie
from scenes import UltimateScene


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        player_labels = dict(o1="A", o2="B", o3="C")
        [f.get_player(name).set_label_text(label) for name, label in player_labels.items()]
        self.write("Counter für trap mark", animate=False)
        self.write_small("ganzes play")
        f.transition(s[1], run_time=0.5)
        f.fake("o2")
        f.transition(s[2], run_time=0.7, linear_end=True)
        f.transition(s[3], run_time=0.7, linear_start=True, disc_delay=0)
        f.fake("o2", run_time=0.1)
        f.transition(s[4], run_time=0.7, linear_end=True)
        f.transition(s[5], run_time=1, linear_start=True, disc_delay=0)

        f.transition(s[3], run_time=0.2)
        self.write_small("No-around mark -> inside ist offen.\nGute Gelegenheit für lefty Rückhand")
        self.wait(2)
        with f.marker_shadow():
            self.wait(1)
            f.transition(s[4], run_time=0.7, disc_delay=0)
            self.wait()
        self.write_small("continuation auf die break side suchen.")
        f.transition(s[5], run_time=0.7, disc_delay=0)
        self.wait(2)


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
