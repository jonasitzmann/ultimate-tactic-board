from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
import sys
import os
import cfg
import importlib
from glob import glob
from run import state_from_photo
from kivy.core.image import Image as CoreImage
from kivy.uix.image import Image as kiImage
from io import BytesIO
import numpy as np
from kivy.uix.label import Label
from kivy.app import App
from kivy.uix.behaviors import DragBehavior
from kivy.lang import Builder
from kivy.uix.relativelayout import RelativeLayout
from kivy.core.window import Window
from cfg import field_height_m, field_width_m
from state import State, Player
import manim_animations
from contextlib import contextmanager


@contextmanager
def cd(path):
    old_path = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(old_path)


Window.size, player_width = (1800, 800), 40
kv = '''
<PlayerWidget>:
    # Define the properties for the DragLabel
    drag_rectangle: self.x, self.y, self.width, self.height
    drag_timeout: 10000000
    drag_distance: 0
    size_hint: (None, None)
    canvas.before:
        Color:
            rgba: 1, 0, 0, 0
        Ellipse:
            pos: self.pos
            size: self.size 
    Label:
        text: root.label
        pos: root.pos
        size: root.size
BoxLayout:
    orientation: 'vertical'
    BoxLayout:
        height: 100
        size_hint: 1, None
        Button:
            text: 'Take Picture'
            on_press: field.take_picture()
        Label:
            text: 'edit pos'
        CheckBox:
            group: 'ckbx'
            on_active: field.edit_players_mode()
        GridLayout:
            cols: 2
            rows: 2
            Label:
                text: 'add offense'
            CheckBox:
                group: 'ckbx'
                on_active: field.add_offenders_mode()
            Label:
                text: 'add defense'
            CheckBox:
                group: 'ckbx'
                on_active: field.add_defenders_mode()
        Button:
            text: 'Save'
            on_press: field.save_state()
        Button:
            text: 'Previous Frame'
            on_press: field.load_frame(field.frame_number - 1)
        Label:
            id: frame_number_label
            text: '1'
        Button:
            text: 'Next Frame'
            on_press: field.load_frame(field.frame_number + 1)
        Button:
            text: 'render'
            on_press: field.render()
    Field:
        id: field
        frame_number_label: frame_number_label  # todo: this hurts
        size_hint: None, None
        height: 670
        width: 1800
    
'''


def get_play_dir():
    os.makedirs(cfg.plays_dir, exist_ok=True)
    dirs = [d.split('/')[-2] for d in glob(f'{cfg.plays_dir}/*/')]
    max_dir = max([int(d) for d in dirs if d.isnumeric()] or [0]) if dirs else 0
    play_number = max_dir + 1
    new_dir = f'{cfg.plays_dir}/{play_number}'
    os.makedirs(new_dir)
    return new_dir, play_number



class Field(RelativeLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.play_dir, self.play_number = get_play_dir()
        self.size_hint = None, None
        self.state = State()
        self.state_img = manim_animations.StateImg()
        self.size_hint = (None, None)
        self.frame_number = 1
        self.current_player_role = None
        self.current_player = None
        self.anglemode = False
        self.play_name = None
        with self.canvas:
            self.image = kiImage(size_hint=(None, None))
        self.reset()
        self.edit_players_mode()
    @property
    def w(self):
        return self.image.width

    @property
    def h(self):
        return self.image.height

    @property
    def num_frames(self):
        return len(glob(f'{self.play_dir}/*.yaml'))

    def prepare_animation_script(self):
        with open(cfg.template_play_file, 'r') as f:
            template = f.read()
        if self.play_name is None:
            self.play_name = f'play{self.play_number}'
        template = template.replace('TemplateScene', self.play_name.capitalize())
        template = template.replace('play_name', self.play_name)
        indentation = ' ' * 8
        transitions = '\n'.join([f'{indentation}f.transition(s[{i}], run_time=4)' for i in range(1, self.num_frames)])
        template = template.replace(f'{indentation}# state transitions', transitions)
        save_path = f'{self.play_dir}/{self.play_name}.py'
        with open(save_path, 'w') as f:
            f.write(template)
        return save_path

    def execute_animation_script(self, script_path):
        sys.path.append(self.play_dir)
        try:
            script = importlib.import_module(f'{self.play_name}')
            with cd(self.play_dir):
                script.render_scene()
        except ImportError:
            print('animation script not found')

    def render(self):
        save_path = self.prepare_animation_script()
        self.execute_animation_script(save_path)

    def on_touch_down(self, touch):
        super().on_touch_down(touch)
        if self.current_player_role is None:
            return
        if not self.collide_point(*touch.pos):
            return
        pos = self.pix2pos(touch.x, touch.y)
        plist = self.state.players_team_1 if self.current_player_role == 'o' else self.state.players_team_2
        label = str(max([int(p.label) for p in plist]) + 1) if plist else '1'
        player = Player(pos, label=label)
        self.current_player = PlayerWidget(player, self)
        plist.append(player)
        self.update_img()

    def on_touch_up(self, touch):
        super().on_touch_up(touch)
        if self.current_player_role is None:
            return
        if self.current_player is None:
            return
        if self.current_player.collide_point(*touch.pos):
            return
        pos1 = self.current_player.player_state.pos
        pos2 = self.pix2pos(touch.x, touch.y)
        self.current_player.player_state.angle = int(np.arctan2(*(pos2 - pos1)) * 180 / np.pi + 180)
        self.update_img()

    def save_state(self):
        self.state.save(f'{self.play_dir}/{self.frame_number}.yaml')
        self.load_frame(self.frame_number + 1)
        self.edit_players_mode()

    def add_players(self):
        for p in self.state.players_team_1 + self.state.players_team_2:
            self.add_widget(PlayerWidget(p, self))

    def load_frame(self, frame_number):
        self.frame_number = max(frame_number, 1)
        self.frame_number_label.text = str(self.frame_number)
        filename = f'{self.play_dir}/{self.frame_number}.yaml'
        if os.path.exists(filename):
            self.state = State.load(filename)
        self.reset()

    def reset(self):
        self.clear_widgets()
        self.add_widget(self.update_img())

    @property
    def scale(self):
        return max([self.h, self.w]) / field_height_m

    def update_img(self):
        img_data = self.state_img.get_img(self.state)
        data = BytesIO()
        img_data.save(data, format='png')
        data.seek(0)  # yes you actually need this
        im = CoreImage(BytesIO(data.read()), ext='png')
        self.image.texture = im.texture
        self.image.size = img_data.size
        return self.image

    def take_picture(self):
        try:
            self.state = state_from_photo()
            self.edit_players_mode()
        except Exception as e:
            print(f'error\n{e}')

    def edit_players_mode(self):
        self.current_player_role = None
        self.reset()
        self.add_players()

    def add_offenders_mode(self):
        self.reset()
        self.current_player_role = 'o'

    def add_defenders_mode(self):
        self.reset()
        self.current_player_role = 'd'

    def pix2pos(self, x, y, offset=0):
        x, y = y + offset, self.w - x - offset
        pos = np.array([x, y]) / self.scale
        pos[0] = field_width_m - pos[0]
        return pos


class PlayerWidget(DragBehavior, Label):
    def __init__(self, player, field, *args, **kwargs):
        self.field, self.player_state = field, player
        self.label = player.label
        self.angle_mode = False
        pos = self.pos2pix(player.pos)
        super().__init__(text=self.label, pos=pos, *args, **kwargs)
        self.width, self.height = player_width, player_width

    def on_touch_down(self, touch):
        if touch.button != 'right':
            return super().on_touch_down(touch)
        if self.collide_point(*touch.pos):
            self.angle_mode = True

    def on_touch_up(self, touch):
        if self.collide_point(touch.x, touch.y):
            self.player_state.pos = self.pix2pos()
            self.parent.update_img()
        elif self.angle_mode:
            pos1 = self.player_state.pos
            pos2 = self.field.pix2pos(touch.x, touch.y)
            self.player_state.angle = int(np.arctan2(*(pos2 - pos1)) * 180 / np.pi + 180)
            self.parent.update_img()
            self.angle_mode = False
        self.angle_mode = False
        if self.collide_point(touch.x, touch.y):
            self.player_state.pos = self.pix2pos()
            self.parent.update_img()
        return super().on_touch_up(touch)

    def pix2pos(self):
        offset = player_width / 2
        return self.field.pix2pos(self.x, self.y, offset)

    def pos2pix(self, pos):
        x, y = pos
        x = field_width_m - x
        x, y = field_height_m - y, x
        return (np.array([x, y]) * self.field.scale - (player_width / 2)).astype(int).tolist()


class UltimateTacticsBoardApp(App):
    def build(self):
        return Builder.load_string(kv)


if __name__ == '__main__':
    UltimateTacticsBoardApp().run()
