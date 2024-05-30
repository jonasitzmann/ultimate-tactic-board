from manim_animations import create_movie
from scenes import UltimateScene
from scenes.ultimate_scene import multi_context


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        f.load_state(s[2])
        self.add(f.get_disc_arrow(s[2], s[3], buffer=0))
        self.add(f.get_disc_arrow(s[3], s[4], buffer=0))
        self.add(f.get_arrows(s[2], s[3], players="o7"))
        self.add(f.get_arrows(s[2], s[4], players="o5"))
        self.add(f.get_arrows(s[3], s[4], players="o1 o7"))


class Animation(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        self.write("Standard Play", animate=False)
        self.write_small("Schnelldurchlauf", animate=False)
        # self.write_small('Wichtig: Timing der Breakside Continuation', animate=False)
        # Durchlauf
        self.wait()
        f.transition(s[1], run_time=0.4)
        f.transition(s[2], run_time=0.5, linear_end=True)
        f.transition(s[3], run_time=1.5, linear_start=True)
        f.transition(s[4], run_time=1.5, disc_delay=0.0, d_delay=0)
        self.wait(2)
        self.write_small("Positionierung")
        f.transition(s[0], run_time=0.5)
        self.wait()
        self.write_small("Handler hinter der Scheibe, um Würfe nach vorne zu ermöglichen")
        with multi_context(f.highlight("o5", play_now=False), f.highlight("o7", play_now=False)):
            self.play(*self.buffered_animations, run_time=0.5)
            self.buffered_animations.clear()
            self.wait()
        self.write_small("Dump auf der offenen Seite wird aktiviert")
        with f.field_of_view("o6"):
            self.wait()
            f.transition(s[1], run_time=0.5)
            self.wait()
            self.write_small("Dump Cuttet auf die Breakside.")
            with f.arrows(players="o7", state=s[1], next_state=s[2]), f.arrows(
                players="o7", state=s[2], next_state=s[3]
            ):
                self.wait()
                self.write_small("Breakside Handler cleart per Überläufer den Raum")
                with f.arrows(players="o5", state=s[1], next_state=s[2]), f.arrows(
                    players="o5", state=s[2], next_state=s[3]
                ):
                    self.wait(2)
                    f.transition(s[2], run_time=0.5, linear_end=True)
                    f.transition(s[3], run_time=1.5, linear_start=True)
            self.wait()
        self.write_small('Iso läuft Breakside continuation. "Late or on time"')
        self.wait()
        f.transition(s[4], run_time=1.5, linear_start=True, d_delay=0, disc_delay=0)
        self.wait()


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    # create_movie(Animation, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
