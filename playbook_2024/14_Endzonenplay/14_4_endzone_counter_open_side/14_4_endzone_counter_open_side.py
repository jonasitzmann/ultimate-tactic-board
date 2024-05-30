from manim_animations import create_movie
from scenes import UltimateScene


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        self.write("Open Side Give'n Go 2 (Counter)", animate=False)
        self.write_small("Falls Dump-Defender die Break Continuation verhindert", animate=False)
        # Durchlauf
        self.wait()
        f.transition(s[1], run_time=0.3)
        self.wait(0.5)
        f.transition(s[2], run_time=0.3, linear_end=True)
        f.transition(s[3], run_time=1, linear_start=True, linear_end=True)
        f.transition(s[4], run_time=1, linear_start=True)
        f.fake(target_player_name="o1")
        f.transition(s[4], run_time=0.4)
        f.transition(s[5], run_time=0.6)
        f.transition(s[6], run_time=0.6, linear_end=True, disc_delay=0.0)
        f.transition(s[7], run_time=1, linear_start=True, disc_delay=0.1)
        self.wait(2)


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
