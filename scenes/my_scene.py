from .ultimate_scene import UltimateScene


class MyScene(UltimateScene):
    def construct(self):
        f, s = self.prepare('../example_play')
        self.call(self.tex('measure distances'), f.measure_distance('o2', 'o4'), t=2)
        self.call(self.tex('measure angles'), f.measure_angle('o2', 'o3'), t=2)
        with self.tex('highlight players:'):
            [self.call(f.highlight(p)) for p in 'o1 o4 d2 d5'.split()]
        self.call(self.tex('add marker shadow'), f.marker_shadow(), t=2)
        self.call(self.tex("show player's field of view"), f.field_of_view('o1'), t=2)
        with self.tex('animate everything'), f.highlight('o4'), f.marker_shadow(),\
                f.field_of_view('o1'), f.measure_distance('o4', 'd4'), f.measure_angle('o2', 'o3'):
            f.transition(next(s))
            self.wait(2)
