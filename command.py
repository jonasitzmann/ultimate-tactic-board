#  https://en.wikipedia.org/wiki/Command_pattern
from abc import ABC, abstractmethod
from player_widget import PlayerWidget
from copy import deepcopy
from functools import partial


class Command(ABC):
    def __init__(self, field):
        self.field = field

    @property
    def state(self):
        return self.field.state

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass


class MovePlayer(Command):
    def __init__(self, field, player, pos, angle):
        super().__init__(field)
        self.old_player, self.new_player = deepcopy(player), deepcopy(player)
        if pos is not None:
            self.new_player.pos = pos
        if angle is not None:
            self.new_player.angle = angle

    def execute(self):
        self.state.set_player(self.new_player)

    def undo(self):
        self.state.set_player(self.old_player)


class AddPlayer(Command):
    def __init__(self, field, role, pos, angle=0):
        super().__init__(field)
        self.role, self.pos, self.angle = role, pos, angle
        self.player_widget = None

    def execute(self):
        player = self.state.add_player(self.role, self.pos, self.angle)
        self.player_widget = PlayerWidget(player, self.field)
        self.field.add_widget(self.player_widget)
        self.field.current_player = self.player_widget

    def undo(self):
        self.field.remove_widget(self.player_widget)
        self.state.remove_player(self.player_widget.player_state)


class AnnotationCommand(Command, ABC):
    def __init__(self, field, annotation):
        super().__init__(field)
        self.annotation = annotation

    def execute(self):
        self.field.configuration.annotations.append(self.annotation)

    def undo(self):
        self.field.configuration.annotations.remove(self.annotation)


class HighlightPlayer(AnnotationCommand):
    def __init__(self, field, annotation):
        from manim_animations import Field as MField
        self.annotation = partial(MField.highlight, player_name=self.player.name)
        super().__init__(field, annotation)


class FieldOfView(AnnotationCommand):
    def __init__(self, field, player):
        from manim_animations import Field as MField
        annotation = partial(MField.field_of_view, player_name=player.name)
        super().__init__(field, annotation)
