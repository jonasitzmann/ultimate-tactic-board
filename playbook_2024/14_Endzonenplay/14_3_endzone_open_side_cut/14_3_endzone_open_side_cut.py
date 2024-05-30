from manim_animations import create_movie
from scenes import UltimateScene


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        self.write("Endzone: Dump-Cut auf die offene Seite", animate=False)


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        self.write("Endzone: Dump-Cut auf die offene Seite", animate=False)
        self.write_small("Falls Dump-Defender den initialen Breakpass verhindert", animate=False)
        # Durchlauf
        self.wait()
        f.transition(s[1], run_time=0.3)
        self.wait(0.5)
        f.transition(s[2], run_time=0.3, linear_end=True)
        f.transition(s[3], run_time=1, linear_start=True)
        f.transition(s[4], run_time=1.5)
        self.wait(2)


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
