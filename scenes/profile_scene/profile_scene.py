import sys
from scenes import UltimateScene
from manim_animations import create_movie
from contextlib import contextmanager
import cProfile
import os


def main():
    create_movie(ProfileScene, debug=True, hq=True, opengl=False)


@contextmanager
def profile(filename=None, *args, **kwargs):
    profiler = cProfile.Profile(*args, **kwargs)
    profiler.enable()
    yield
    profiler.disable()
    if filename:
        profiler.dump_stats(os.path.expanduser(filename))
    profiler.print_stats(sort=1)


class ProfileScene(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        f.transition(s[1], run_time=5)



if __name__ == '__main__':
    main()