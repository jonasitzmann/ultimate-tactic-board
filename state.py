import numpy as np
import attr_dict
import cfg
import yaml
from itertools import chain
from contextlib import contextmanager


class Player:
    def __init__(self, pos, angle=0, label='', color=None, role=''):
        self.pos = np.array(pos)
        self.angle = int(angle)
        self.label = label
        self.color = np.array(color)
        self.role = role

    @property
    def name(self):
        return f'{self.role}{self.label}'

    @staticmethod
    def from_dict(d):
        d = attr_dict.AttrDict(d)
        return Player(pos=d.pos, angle=d.orientation, label=d.label, color=d.color, role=d.role)

    @property
    def manimpos(self):
        return np.array([cfg.field_width_m - self.pos[0], self.pos[1], 1])

    @property
    def manimangle(self):
        return np.deg2rad(self.angle - 180)

    def __repr__(self):
        return str(self.__dict__)

    def __lt__(self, other):
        return self.label < other.label


class State:
    """
    represents the state of play at an instant.
    """
    def __init__(self, players=None, disc=None, areas=None):
        self.players = players if players is not None else {'o': {}, 'd': {}}
        self.disc = [] if disc is None else disc
        self.areas = [] if areas is None else areas

    def __repr__(self):
        return str(self.__dict__)

    def add_player(self, role, pos, angle=0):
        label = str(len(self.players[role]) + 1)
        player = Player(pos, angle, label, role=role)
        self.set_player(player)
        return player

    def remove_player(self, player):
        if player.role not in self.players:
            return
        self.players[player.role].pop(player.label)

    @property
    def playerlist(self):
        return chain(*[pdict.values() for pdict in self.players.values()])

    def get_player(self, player):
        if player is not None:
            return self.players.get(player.role, {}).get(player.label, None)

    def set_player(self, player):
        if player.role not in self.players:
            self.players[player.role] = {}
        self.players[player.role][player.label] = player

    def find_player(self, name):
        role, label = name[0], name[1:]
        return self.players.get(role, {}).get(label, None)

    def find_players(self, names):
        return [self.find_player(name) for name in names.split(' ')]

    def align_x(self, players, to='mean'):
        players = self.find_players(players)
        if to == 'mean':
            x = np.mean([p.pos[0] for p in players])
        elif to == 'min':
            x = min([p.pos[0] for p in players])
        elif to == 'max':
            x = max([p.pos[0] for p in players])
        else:
            player = self.find_player(to)
            if player is None:
                print(f'player {to} not found')
                return
            if player not in players:
                players.append(player)
            x = player.pos[0]
        for p in players:
            p.pos[0] = x

    @contextmanager
    def transpose_poses(self, players):
        players = self.find_players(players)
        for p in players:
            p.pos = p.pos[::-1]
        yield
        for p in players:
            p.pos = p.pos[::-1]

    def align_y(self, players, to='mean'):
        with self.transpose_poses(players):
            self.align_x(players, to)

    @staticmethod
    def from_dict(d):
        players = d['players']
        for role, player_dict in players.items():
            for label, player in player_dict.items():
                player_dict[label] = Player.from_dict(player)
        d = attr_dict.AttrDict(d)
        return State(
            players=players,
            disc=d.disc,
            areas=d.areas
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
        players=state.players,
        disc=np.array(state.disc).reshape(-1).tolist(),
        areas=state.areas
    ))


def players_representer(dumper: yaml.Dumper, player: Player):
    return dumper.represent_dict(dict(
        pos=player.pos.tolist(),
        orientation=player.angle,
        color=player.color.tolist(),
        label=player.label,
        role=player.role,
    ))

yaml.add_representer(State, state_representer)
yaml.add_representer(Player, players_representer)

if __name__ == "__main__":
    s = State([Player([1, 2], 3, '4', (5, 6, 7))], [], [8, 9], [[10, 11, 12, 10]])
    filename = cfg.temp_dir + 's.yaml'
    s.save(filename)
    s2 = State.load(filename)
    print(s2)

