import numpy as np
import os
import attr_dict
import cfg
import yaml


class Player:
    def __init__(self, pos, angle=0, label='', color=None):
        self.pos = np.array(pos)
        self.angle = int(angle)
        self.label = label
        self.color = np.array(color)

    @staticmethod
    def from_dict(d):
        d = attr_dict.AttrDict(d)
        return Player(
            pos=d.pos,
            angle=d.orientation,
            label=d.label,
            color=d.color
        )

    @property
    def manimpos(self):
        return np.array([cfg.field_width_m - self.pos[0], self.pos[1], 1])

    def __add__(self, other):
        assert(other.label == self.label)
        return Player(self.pos + other.pos, self.angle + other.angle, self.label, self.color)

    def __mul__(self, x):
        return Player(self.pos * x, self.angle * x, self.label, self.color)

    def __lt__(self, other):
        return self.label < other.label

    def __repr__(self):
        return self.__dict__

    __rmul__ = __mul__


class State:
    """
    represents the state of play at an instant.
    The functions __add__ and __mul__ allow for addition and scalar multiplication
    which is required for interpolation.
    # todo: should the interpolation really be done this way?
    # todo: There are several possible ways to interpolate beween states.
    """
    def __init__(self, players_team_1=None, players_team_2=None, disc=None, areas=None):
        self.players_team_1 = sorted(players_team_1 or [])
        self.players_team_2 = sorted(players_team_2 or [])
        self.disc = np.array(disc)
        self.areas = np.array(areas or [])

    def __add__(self, other):
        red_players = [s + o for s, o in zip(self.players_team_1, other.players_team_1)]
        blue_players = [s + o for s, o in zip(self.players_team_2, other.players_team_2)]
        disc = self.disc + other.disc
        return State(players_team_1=red_players, players_team_2=blue_players, areas=other.areas, disc=disc)

    def __mul__(self, x):
        red_players = [s * x for s in self.players_team_1]
        blue_players = [s * x for s in self.players_team_2]
        disc = self.disc * x
        return State(players_team_1=red_players, players_team_2=blue_players, areas=self.areas, disc=disc)

    @staticmethod
    def from_dict(d):
        d = attr_dict.AttrDict(d)
        return State(
            players_team_1=[Player.from_dict(p) for p in d.players_team_1],
            players_team_2=[Player.from_dict(p) for p in d.players_team_2],
            disc=np.array(d.disc),
            areas=np.array(d.areas)
        )

    def save(self, path):
        yaml.add_representer(State, state_representer)
        yaml.add_representer(Player, players_representer)
        with open(path, 'w') as f:
            yaml.dump(self, f, default_flow_style=None)
        pass

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return State.from_dict(d)


def state_representer(dumper: yaml.Dumper, state: State):
    return dumper.represent_dict(dict(
        players_team_1=state.players_team_1,
        players_team_2=state.players_team_2,
        disc=state.disc.tolist(),
        areas=[a.tolist() for a in state.areas]
    ))


def players_representer(dumper: yaml.Dumper, player: Player):
    return dumper.represent_dict(dict(
        pos=player.pos.tolist(),
        orientation=player.angle,
        color=player.color.tolist(),
        label=player.label
    ))

yaml.add_representer(State, state_representer)
yaml.add_representer(Player, players_representer)

if __name__ == "__main__":
    s = State([Player([1, 2], 3, '4', (5, 6, 7))], [], [8, 9], [[10, 11, 12, 10]])
    filename = cfg.temp_dir + 's.yaml'
    s.save(filename)
    s2 = State.load(filename)
    print(s2)

