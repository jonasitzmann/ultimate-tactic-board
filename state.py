from dataclasses import dataclass
import numpy as np


class Player:
    def __init__(self, pos, orientation=0, label='', color=None):
        self.pos = np.array(pos)
        self.orientation = orientation
        self.label = label
        self.color = color

    def __add__(self, other):
        assert(other.label == self.label)
        return Player(self.pos + other.pos, self.orientation + other.orientation, self.label, self.color)

    def __mul__(self, x):
        return Player(self.pos * x, self.orientation * x, self.label, self.color)

    def __lt__(self, other):
        return self.label < other.label

    __rmul__ = __mul__


class State:
    def __init__(self, players_team_1, players_team_2, disc=None, areas=None):
        self.players_team_1 = sorted(players_team_1)
        self.players_team_2 = sorted(players_team_2)
        self.disc = disc
        self.areas = areas

    def __add__(self, other):
        red_players = [s + o for s, o in zip(self.players_team_1, other.players_team_1)]
        blue_players = [s + o for s, o in zip(self.players_team_2, other.players_team_2)]
        disc = self.disc + other.disc
        # todo handle disc position
        return State(players_team_1=red_players, players_team_2=blue_players, areas=other.areas, disc=disc)

    def __mul__(self, x):
        red_players = [s * x for s in self.players_team_1]
        blue_players = [s * x for s in self.players_team_2]
        disc = self.disc * x
        # todo handle disc position
        return State(players_team_1=red_players, players_team_2=blue_players, areas=self.areas, disc=disc)

