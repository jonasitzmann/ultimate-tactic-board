#  https://en.wikipedia.org/wiki/Command_pattern
from abc import ABC, abstractmethod


class Command(ABC):
    def __init__(self, field):
        self.field = field

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass


class MovePlayer(Command):
    def __init__(self, field, player, pos, angle):
        super().__init__(field)
        self.old_player = player.copy()
        self.new_player = player.copy()
        self.new_player.pos = pos
        self.new_player.angle = angle

    def execute(self):
        self.field.state.get_player(self.old_player)
        self.field.state. = self.new_player

    def undo(self):
        self.field.players[self.new_player.id] = self.old_player