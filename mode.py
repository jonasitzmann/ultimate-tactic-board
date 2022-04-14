from abc import ABC

from kivy.graphics import Rectangle, Color
from kivy.properties import ObjectProperty
from kivy.uix.behaviors import DragBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget

import cfg
from player_widget import PlayerWidget
import command
import numpy as np


class SelectionRect(Widget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p1 = self.pos.copy()
        self.p2 = self.pos.copy()
        self.set_p2(self.p2)

    def set_p2(self, p2):
        self.p2 = p2
        self.x = min(self.p1[0], self.p2[0])
        self.y = min(self.p1[1], self.p2[1])
        self.width = abs(self.p1[0] - self.p2[0])
        self.height = abs(self.p1[1] - self.p2[1])
        return


class Mode(ABC):
    def __init__(self, field, get_widgets_func=None):
        self.get_widgets = get_widgets_func or self.get_players
        self.field = field
        field.reset()
        self.widgets = []
        self.add_widgets()

    def add_widgets(self):
        self.widgets = self.get_widgets()
        for w in self.widgets:
            self.field.add_widget(w)

    def remove_widgets(self):
        for widget in self.widgets:
            self.field.remove_widget(widget)

    def on_touch_down(self, touch):
        pass

    def on_touch_up(self, touch):
        pass

    def on_touch_move(self, touch):
        pass

    def reload(self):
        self.remove_widgets()
        self.add_widgets()
        pass

    def get_widget_at(self, pos, widget_list=None):
        if widget_list is None:
            widget_list = self.widgets
        for widget in widget_list:
            if widget.collide_point(*pos):
                return widget

    def get_players(self):
        return [PlayerWidget(p, self.field) for p in self.field.state.players]

    def __del__(self):
        self.remove_widgets()


class EditPoseMode(Mode):
    def __init__(self, field):
        super().__init__(field)
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
                    move_distance = np.linalg.norm(pos - prev_player.pos)
                    if move_distance > max_distance_no_turn:
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
    def __init__(self, field):
        super().__init__(field)
        self.current_player = None

    def on_touch_down(self, touch):
        role = 'd' if self.field.ctrl_pressed else 'o'
        if not self.field.collide_point(*touch.pos):
            return
        pos = self.field.pix2pos(*touch.pos)
        if touch.button == 'left':
            cmd = command.AddPlayer(self.field, role, pos)
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
        # self.selection_commands = []
        self.selected_players = []
        self.selection_rect = None

    def __del__(self):
        super().__del__()
        self.undo_annotations()

    def undo_annotations(self):
        # for cmd in self.selection_commands:
        #     self.field.do_and_reload(cmd.undo)
        # self.selection_commands = []
        self.selected_players = []

    def on_touch_down(self, touch):
        self.undo_annotations()
        if touch.button == 'left':
            player = self.get_widget_at(touch.pos)
            if isinstance(player, PlayerWidget):
                # cmd = command.FieldOfView(self.field, player.player_state)
                self.selected_players = [player.player_state]
                # self.field.do_and_reload(cmd.execute)
                # self.selection_commands.append(cmd)
            else:
                self.selection_rect = SelectionRect(pos=touch.pos)
                self.widgets.append(self.selection_rect)
                self.field.add_widget(self.selection_rect)

    def on_touch_move(self, touch):
        if self.selection_rect is not None:
            self.selection_rect.set_p2(touch.pos)

    def on_touch_up(self, touch):
        if self.selection_rect is not None:
            self.selected_players = [
                p.player_state for p in self.widgets
                if self.selection_rect.collide_widget(p) and p is not self.selection_rect]
            self.field.remove_widget(self.selection_rect)
            self.widgets.remove(self.selection_rect)
            self.selection_rect = None
        if self.selected_players:
            self.field.mode = PlayersSelectedMode(self.field, self.selected_players)


class SelectionMenu(BoxLayout):
    mode = ObjectProperty(None)


class PlayersSelectedMode(Mode):
    def __init__(self, field, players):
        self.players = players
        super().__init__(field, self.get_widgets)
        self.drag = False
        self.menu = SelectionMenu(mode=self)
        self.field.parent.add_widget(self.menu, index=1)
        self.highlight_cmd = None
        self.fov_annotations = None


    def __del__(self):
        self.field.parent.remove_widget(self.menu)
        super().__del__()

    def get_widgets(self):
        player_states = [self.field.state.get_player(p) for p in self.players]
        return [PlayerWidget(p, self.field) for p in player_states]

    def on_touch_down(self, touch):
        if self.get_widget_at(touch.pos) is None:
            self.field.mode = SelectMode(self.field)
            self.field.mode.on_touch_down(touch)
        else:
            self.drag = True

    def on_touch_move(self, touch):
        if self.drag:
            for widget in self.widgets:
                widget.x += touch.dx
                widget.y += touch.dy

    def on_touch_up(self, touch):
        if self.drag:
            cmds = []
            for player_widget in self.widgets:
                cmds.append(command.MovePlayer(self.field, player_widget.player_state, pos=player_widget.pix2pos()))
            self.field.execute_cmd(command.CommandList(cmds))
            self.drag = False

    def align_x(self):
        self.field.do_and_reload(lambda: self.field.state.align_x(' '.join([p.name for p in self.players])))

    def align_y(self):
        self.field.do_and_reload(lambda: self.field.state.align_y(' '.join([p.name for p in self.players])))

    def highlight_toggle(self):
        if self.highlight_cmd is None:
            self.highlight_cmd = command.HighlightPlayers(self.field, *self.players)
            self.field.do_and_reload(self.highlight_cmd.execute)
        else:
            self.field.do_and_reload(self.highlight_cmd.undo)
            self.highlight_cmd = None
        self.field.do_and_reload(lambda: None)

    def fov_toggle(self):
        if self.fov_annotations is None:
            self.fov_annotations = command.FieldOfView(self.field, *self.players)
            self.field.do_and_reload(self.fov_annotations.execute)
        else:
            self.field.do_and_reload(self.fov_annotations.undo)
            self.fov_annotations = None
        self.field.do_and_reload(lambda: None)










class ViewMode(Mode):
    pass
