from manim_animations import create_movie
from scenes import UltimateScene


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        self.write("Inside Break ohne Mark", animate=False)
        self.write_small("Falls Mark overcommittet.", animate=False)
        # Durchlauf
        self.wait()
        f.transition(s[1], run_time=0.3)
        self.wait(0.5)
        f.transition(s[2], run_time=0.3, linear_end=True)
        f.transition(s[3], run_time=1.5, linear_start=True)
        f.fake(target_player_name="o7")
        f.transition(s[4], run_time=0.5)
        f.transition(s[5], run_time=0.5)
        self.wait(2)


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
