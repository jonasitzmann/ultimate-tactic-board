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
    def __init__(self, field, player, pos=None, angle=None):
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


class AnnotationsCommand(Command, ABC):
    def __init__(self, field, annotations):
        super().__init__(field)
        self.annotations = annotations

    def execute(self):
        for annotation in self.annotations:
            self.field.configuration.annotations.append(annotation)

    def undo(self):
        for annotation in self.annotations:
            self.field.configuration.annotations.remove(annotation)


class HighlightPlayers(AnnotationsCommand):
    def __init__(self, field, *players):
        from manim_animations import Field as MField
        annotations = [partial(MField.highlight, player_name=player.name) for player in players]
        super().__init__(field, annotations)


class FieldOfView(AnnotationsCommand):
    def __init__(self, field, *players):
        from manim_animations import Field as MField
        annotations = [partial(MField.field_of_view, player_name=player.name) for player in players]
        super().__init__(field, annotations)


class CommandList(Command):
    def __init__(self, commands):
        self.commands = commands
        super().__init__(None)

    def execute(self):
        for command in self.commands:
            command.execute()

    def undo(self):
        for command in reversed(self.commands):
            command.undo()
