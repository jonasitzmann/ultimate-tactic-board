import numpy as np
import attr_dict
import cfg
import yaml
from contextlib import contextmanager
import tactics_utils


class Player:
    def __init__(self, pos=None, angle=0, label='', role=''):
        self.pos = np.array(pos)
        self.angle = int(angle)
        self.label = label
        self.role = role

    @property
    def name(self):
        return f'{self.role}{self.label}'

    @staticmethod
    def from_dict(d):
        d = attr_dict.AttrDict(d)
        return Player(pos=d.pos, angle=d.orientation, label=d.label, role=d.role)

    @property
    def manimpos(self):
        return np.array([self.pos[0], self.pos[1], 1])

    @property
    def manimangle(self):
        return np.deg2rad(self.angle)

    def __repr__(self):
        return str(self.__dict__)

    def __lt__(self, other):
        return self.label < other.label


class State:
    """
    represents the state of play at an instant.
    """
    def __init__(self, players=None, disc=None, areas=None):
        self.players = players if players is not None else []
        self.disc = [] if disc is None else disc
        self.areas = [] if areas is None else areas

    def __repr__(self):
        return str(self.__dict__)

    def add_player(self, role, pos, angle=0):
        labels = [p.label for p in self.players if p.role == role]
        label = str(min([x for x in range(len(labels) + 1) if str(x + 1) not in labels]) + 1)
        player = Player(pos, angle, label, role=role)
        self.players.append(player)
        return player

    def remove_player(self, player):
        if player in self.players:
            self.players.remove(player)

    def get_player(self, reference_player):
        player = None
        if reference_player is not None:
            matches = [p for p in self.players if p.name == reference_player.name and p.role == reference_player.role]
            if len(matches) == 1:
                player = matches[0]
            else:
                print(f'possible players: {matches}')
        return player

    def set_player(self, player):
        self.remove_player(self.get_player(player))
        self.players.append(player)

    def find_player(self, name):
        role, label = name[0], name[1:]
        return self.get_player(Player(role=role, label=label))

    def find_players(self, names):
        return [self.find_player(name) for name in names.split(' ')]

    def get_disc_holder(self):
        offenders = [p for p in self.players if p.role == 'o']
        dists = [np.linalg.norm(p.pos - self.disc) for p in offenders]
        return offenders[np.argmin(dists)]

    def setup_hex(self, pos=None, angle=None, dist=None, min_sideline_dist=None):
        pos = pos if pos is not None else np.array([cfg.field_width_m/2, 20])
        dist = dist if dist is not None else cfg.hex_dist_m
        sin_60 = np.sqrt(3) / 2
        if angle is None:
            angle = tactics_utils.get_hex_angle(pos, dist, min_sideline_dist)
        self.disc = np.array(pos)
        self.players.clear()
        pts = np.array([[0, 0], [0, 1], [0, 2], [sin_60, 0.5], [-sin_60, 0.5], [sin_60, 1.5], [-sin_60, 1.5]]).T * dist
        angle = np.deg2rad(angle)
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        pts = (rot_matrix @ pts) + pos[:, None]
        for p in pts.T:
            self.add_player('o', p)

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
        d = attr_dict.AttrDict(d)
        return State(
            players=[Player.from_dict(player) for player in d.players],
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

