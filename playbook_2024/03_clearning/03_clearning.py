from manim_animations import create_movie
from scenes import UltimateScene
from manim import *


class Thumbnail(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        f.load_state(s[1])
        self.add(f.get_arrows(s[0], s[1], players="o2"))
        good_arrow = f.get_arrows(s[1], s[2], players="o2")[0].set_color(GREEN)
        good_text = Tex("Gut. Anspielbar + Raum wird frei", font_size=20).next_to(good_arrow, UP)
        bad_arrow = f.get_arrows(s[1], s["3_bad"], players="o2")[0].set_color(RED)
        bad_text = Tex(
            r"Schlecht.\\Raum wird blockiert.\\Kein sinnvolles nächstes\\Angebot von hier möglich",
            font_size=20,
        ).next_to(bad_arrow, DL, 0)
        self.add(good_arrow, bad_arrow, good_text, bad_text)


class Play1(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        self.write("Bei kurzem Angebot: Clearing statt Laufduell zur Scheibe", animate=False)
        self.write_small("Alles auf einmal", animate=False)
        self.wait()
        f.transition(s[1], run_time=1)
        good_arrow = f.get_arrows(s[1], s[2], players="o2")[0].set_color(GREEN)
        good_text = Tex("Gut", font_size=20).next_to(good_arrow, UP)
        bad_arrow = f.get_arrows(s[1], s["3_bad"], players="o2")[0].set_color(RED)
        bad_text = Tex(r"Schlecht", font_size=20).next_to(bad_arrow, LEFT)
        good_bad_mobjects = [good_arrow, bad_arrow, good_text, bad_text]
        self.play(FadeIn(Group(*good_bad_mobjects)))
        self.add(*good_bad_mobjects)
        f.transition(s["3_good_bad"], run_time=1)
        self.wait()
        f.transition(s[1], run_time=0.3)
        self.write_small(r"Schlecht: Bis zur Scheibe laufen.\\Selbst wenn man gefühlt frei ist, blockiert man den Raum")
        self.wait()
        f.transition(s["3_bad"], run_time=1)
        self.wait()
        f.transition(s[1], run_time=0.3)
        self.write_small(
            r"""
        Gut: Nach kurzem Angebot auf die geschlossene Seite clearen.\\
        Evtl. inside anspielbar. Vor allem wird aber der Raum frei!"""
        )
        self.wait()
        f.transition(s[2], run_time=1)
        self.wait(2)
        return


def render_scene():
    create_movie(Thumbnail, debug=False, hq=True)
    create_movie(Play1, debug=False, hq=True)


if __name__ == "__main__":
    render_scene()
