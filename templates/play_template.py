from manim_animations import create_movie
from scenes import UltimateScene


class TemplateScene(UltimateScene):
    def construct(self):
        f, s = self.prepare()
        # state transitions


def render_scene():
    create_movie(TemplateScene, debug=False, hq=True, output_file='play_name.mp4')


if __name__ == '__main__':
    render_scene()
