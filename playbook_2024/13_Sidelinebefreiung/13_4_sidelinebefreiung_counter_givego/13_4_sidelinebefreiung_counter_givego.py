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
        self.write("Sidelinebefreiung, Marker overcommitted", animate=False)
        self.write_small("ganzes play")
        self.wait()
        f.transition(s[1], run_time=0.5)
        f.fake("o2")
        f.transition(s[2], run_time=0.7)
        f.transition(s[3], run_time=0.7)
        f.fake("o2", run_time=0.1)
        f.transition(s[4], run_time=0.5, d_delay=0)
        f.transition(s[5], run_time=1.2)
        f.transition(s[6], run_time=0.7, linear_end=True, disc_delay=0)
        f.transition(s[7], run_time=1, linear_start=True, d_delay=0)
        self.wait(1)

        f.transition(s[4], run_time=0.2, d_delay=0)
        self.write_small("marker versucht, breakside continuation zu verhindern\nsteht dadurch weit auf der breakside")
        with f.highlight("d3"):
            self.wait(2)
            f.transition(s[5], run_time=1.2, disc_delay=0)
            self.wait()
            self.write_small("counter: give'n go auf die offene Seite")
            self.wait(2)
        f.transition(s[6], run_time=0.7, linear_end=True, disc_delay=0)
        f.transition(s[7], run_time=1, linear_start=True, d_delay=0)
        self.wait(2)


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
