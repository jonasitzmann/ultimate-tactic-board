from scenes import UltimateScene
from manim_animations import create_movie


class MyScene(UltimateScene):
    def construct(self):
        f, s = self.prepare('../../example_play')
        # self.call(self.tex("show player's field of view"), f.field_of_view('o1'), t=2)
        # self.call(self.tex('measure distances'), f.measure_distance('o2', 'o4'), t=2)
        # self.call(self.tex('measure angles'), f.measure_angle('o2', 'o3'), t=2)
        # with self.tex('highlight players:'):
        #     [self.call(f.highlight(p)) for p in 'o1 o4 d2 d5'.split()]
        # self.call(self.tex('add marker shadow'), f.marker_shadow(), t=2)
        with f.measure_distance('o4', 'o7', fade=False), f.measure_angle('o2', 'o3', fade=False):
            f.transition(s[1])
            self.wait(2)



if __name__ == '__main__':
    create_movie(MyScene, debug=False, hq=True)
