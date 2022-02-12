# from manim_animations import *
# class UltimateScene(Scene):
#     def construct(self):
#         s1, s2 = get_states_from_dir('example_play')
#         field = Field(self, s1)
#         with field.s.offense['5'].highlight(self):
#             with field.s.marker_shadow(self):
#                 field.transition(self, s2)
#         field.landscape_to_portrait(self)
#         text = Text('Code:').to_corner(UL)
#         code = Code(__file__, line_spacing=0.5).next_to(
#             text, DOWN, aligned_edge=UL, buff=1)
#         self.play(Write(text), Write(code))
#         self.wait(5)