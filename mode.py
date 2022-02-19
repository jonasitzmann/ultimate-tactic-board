from abc import ABC

import cfg
from player_widget import PlayerWidget
import command
import numpy as np


class Mode(ABC):
    def __init__(self, field):
        self.field = field
        self.widgets = []
        field.reset()

    def on_touch_down(self, touch):
        pass

    def on_touch_up(self, touch):
        pass

    def on_touch_move(self, touch):
        pass

    def get_widget_at(self, pos):
        for widget in self.widgets:
            if widget.collide_point(*pos):
                return widget

    def add_players(self):
        for pdict in self.field.state.players.values():
            for p in pdict.values():
                player = PlayerWidget(p, self.field)
                self.widgets.append(player)
                self.field.add_widget(player)

    def __del__(self):
        for widget in self.widgets:
            self.field.remove_widget(widget)


class EditPoseMode(Mode):
    def __init__(self, field):
        super().__init__(field)
        self.add_players()
        self.angle_mode = False
        self.current_player = None

    def on_touch_down(self, touch):
        self.current_player = self.get_widget_at(touch.pos)
        if self.current_player is not None:
            if touch.button == 'left':
                self.current_player.on_touch_down(touch)
            else:
                self.angle_mode = True

    def on_touch_up(self, touch):
        if self.current_player is not None:
            angle = pos = None
            if touch.button == 'left':
                self.current_player.on_touch_up(touch)
                pos = self.current_player.pix2pos()
                prev_player = self.field.get_previous_player(self.current_player.player_state)
                if prev_player is not None:
                    max_distance_no_turn = 3
                    if np.linalg.norm(pos - prev_player.pos) > max_distance_no_turn:
                        angle = float(np.arctan2(*(pos - prev_player.pos)) * 180 / np.pi + 180)
            elif self.angle_mode and not self.current_player.collide_point(*touch.pos):
                pos1 = self.current_player.pix2pos()
                pos2 = self.field.pix2pos(*touch.pos)
                angle = int(np.arctan2(*(pos2 - pos1)) * 180 / np.pi + 180)
            if pos is not None or angle is not None:
                cmd = command.MovePlayer(self.field, self.current_player.player_state, pos, angle)
                self.field.execute_cmd(cmd)
        self.current_player = None
        self.angle_mode = False



class AddPlayerMode(Mode):
    def __init__(self, field, role):
        super().__init__(field)
        self.role = role
        self.current_player = None

    def on_touch_down(self, touch):
        if not self.field.collide_point(*touch.pos):
            return
        pos = self.field.pix2pos(*touch.pos)
        if touch.button == 'left':
            cmd = command.AddPlayer(self.field, self.role, pos)
            self.field.execute_cmd(cmd)
            self.current_player = cmd.player_widget

        elif touch.button == 'middle':
            disc_pos = pos
            disc_pos[0] = cfg.field_width_m - disc_pos[0]
            self.field.state.disc = disc_pos
            self.field.update_img()

    def on_touch_up(self, touch):
        if self.current_player is None or self.current_player.player_state is None or self.current_player.collide_point(*touch.pos):
            return
        pos1 = self.current_player.player_state.pos
        pos2 = self.field.pix2pos(touch.x, touch.y)
        angle = int(np.arctan2(*(pos2 - pos1)) * 180 / np.pi + 180)
        cmd = command.MovePlayer(self.field, self.current_player.player_state, None, angle)
        self.field.execute_cmd(cmd)
        self.current_player = None


class SelectMode(Mode):
    def __init__(self, field):
        super().__init__(field)
        self.add_players()
        self.selection_commands = []

    def on_touch_down(self, touch):
        for cmd in self.selection_commands:
            self.field.do_and_reload(cmd.undo)
        self.selection_commands = []
        if touch.button == 'left':
            player = self.get_widget_at(touch.pos)
            if player is not None:
                cmd = command.FieldOfView(self.field, player.player_state)
                self.field.do_and_reload(cmd.execute)
                self.selection_commands.append(cmd)

    def on_touch_up(self, touch):
        pass



class ViewMode(Mode):
    pass
