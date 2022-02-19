from kivy.uix.label import Label
from kivy.uix.behaviors import DragBehavior
import cfg
import numpy as np
player_width, player_height = 40, 40


class PlayerWidget(DragBehavior, Label):
    def __init__(self, player, field, *args, **kwargs):
        self.field, self.player_state_ = field, player
        self.label = player.label
        self.angle_mode = False
        pos = self.pos2pix(player.pos)
        super().__init__(text=self.label, pos=pos, *args, **kwargs)
        self.width, self.height = player_width, player_width

    @property
    def player_state(self):
        return self.field.state.get_player(self.player_state_)


    def pix2pos(self):
        offset = player_width / 2
        return self.field.pix2pos(self.x, self.y, offset)

    def pos2pix(self, pos):
        x, y = pos
        x = cfg.field_width_m - x
        x, y = cfg.field_height_m - y, x
        return (np.array([x, y]) * self.field.scale - (player_width / 2)).astype(int).tolist()
