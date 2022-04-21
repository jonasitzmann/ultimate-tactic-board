from manim import *


class FixArrow(Arrow):
    def __init__(self, start, end, *args, **kwargs):
        stroke_width, tip_size = 5, 0.1
        super().__init__(ORIGIN, RIGHT, *args, **kwargs, stroke_width=stroke_width, max_tip_length_to_length_ratio=tip_size, buff=0)
        end -= 0.05 * normalize(end - start)
        self.put_start_and_end_on(start, end)
        self.set_stroke(width=stroke_width)
        tip = self.create_tip(tip_length=0.18)
        self.remove(self.tip)
        self.add(tip)


class Decision(VMobject):
    def __init__(self, text):
        super().__init__()
        w, h = 2, 0.65
        self.add(RoundedRectangle(0.05, height=1, width=1).rotate(PI/4)).apply_matrix(np.array([[w, 0, 0], [0, h, 0], [0, 0, 1]]))
        self.text = Tex(text).scale(0.48)
        self.add(self.text)


class Action(VMobject):
    def __init__(self, text):
        super().__init__()
        w, h = 2, 0.65
        self.add(Rectangle(height=h, width=w))
        self.text = Tex(text).scale(0.48)
        self.add(self.text)


class Group(VGroup):
    def add(self, *mobjects):
        super().add(*mobjects)
        if len(mobjects) == 1:
            mobjects = mobjects[0]
        return mobjects


def get_focus_mobject(mobject):
    focus_mobject = mobject.copy()
    focus_mobject.remove(focus_mobject.text)
    focus_mobject.set_fill(opacity=1, color=BLUE_E)
    focus_mobject.set_stroke(opacity=0)
    focus_mobject.set_z_index(-1)
    return focus_mobject


focus_mobject = None
def focus(mobject):
    global focus_mobject
    new_focus_mobject = get_focus_mobject(mobject)
    if focus_mobject is None:
        anim = FadeIn(focus_mobject)
    else:
        anim = ReplacementTransform(focus_mobject, new_focus_mobject)
    focus_mobject = new_focus_mobject
    return anim
