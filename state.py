import numpy as np
import attr_dict
import cfg
import yaml


class Player:
    def __init__(self, pos, angle=0, label='', color=None, role=''):
        self.pos = np.array(pos)
        self.angle = int(angle)
        self.label = label
        self.color = np.array(color)
        self.role = role

    @staticmethod
    def from_dict(d):
        d = attr_dict.AttrDict(d)
        return Player(pos=d.pos, angle=d.orientation, label=d.label, color=d.color, role=d.role)

    @property
    def manimpos(self):
        return np.array([cfg.field_width_m - self.pos[0], self.pos[1], 1])

    def __repr__(self):
        return str(self.__dict__)

    def __lt__(self, other):
        return self.label < other.label


class State:
    """
    represents the state of play at an instant.
    """
    def __init__(self, players_team_1=None, players_team_2=None, disc=None, areas=None):
        # todo: why two lists of players?
        # todo: find players, disc(s) and areas by id
        # todo: at least players_team_1 and players_team_2 should be dicts rather than lists
        self.players_team_1 = sorted([] if players_team_1 is None else players_team_1)
        self.players_team_2 = sorted([] if players_team_2 is None else players_team_2)
        self.disc = np.array(disc)
        self.areas = [] if areas is None else areas

    def __repr__(self):
        return str(self.__dict__)

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

