import os
import cfg
from glob import glob
from run import state_from_photo
from PIL import Image as PILImage
import numpy as np
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.app import App
from kivy.uix.behaviors import DragBehavior
from kivy.lang import Builder
from kivy.uix.relativelayout import RelativeLayout
from kivy.core.window import Window
from kivy.properties import ListProperty, StringProperty, NumericProperty

from cfg import field_height_m, field_width_m
from state import State, Player
from explore_manim import StateImg
Window.size = (1500, 650)
player_width = 40
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
        Button:
            text: 'Load'
            on_press: field.load_state()
        Button:
            text: 'Edit'
            on_press: field.edit_players_mode()
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
    Field:
        id: field
        frame_number_label: frame_number_label  # todo: this hurts
        size_hint: None, None
        height: 550
        width: 1500
    
'''


def get_play_dir():
    os.makedirs(cfg.plays_dir, exist_ok=True)
    dirs = [d.split('/')[-2] for d in glob(f'{cfg.plays_dir}/*/')]
    max_dir = max(int(d) for d in dirs if d.isnumeric()) if dirs else 0
    new_dir = f'{cfg.plays_dir}/{max_dir + 1}'
    os.makedirs(new_dir)
    return new_dir


class Field(RelativeLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.play_dir = get_play_dir()
        self.size_hint = None, None
        self.state = State.load(f'temp/s.yaml')
        self.img_source = StateImg.get(self.state)
        self.size_hint = (None, None)
        self.frame_number = 1
        with self.canvas:
            self.w, self.h = PILImage.open(self.img_source).size
            self.image = Image(size_hint=(None, None), height=self.h, width=self.w, source=None)
        self.reset()

    def save_state(self):
        self.state.save(f'{self.play_dir}/{self.frame_number}.yaml')

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

    def pix2pos(self, x, y):
        x, y = y, self.field.w - x
        pos = np.array([x, y]) / self.scale
        pos[0] = field_width_m - pos[0]
        return pos

    def reset(self):
        self.clear_widgets()
        self.add_widget(self.update_img())

    def update_img(self):
        self.image.source = StateImg.get(self.state)
        self.image.reload()
        return self.image

    def take_picture(self):
        try:
            self.state = state_from_photo()
            self.reset()
            self.add_players()
        except Exception as e:
            print(f'error\n{e}')

    def edit_players_mode(self):
        self.reset()
        self.add_players()


class PlayerWidget(DragBehavior, Label):
    def __init__(self, player, field, *args, **kwargs):
        self.field, self.player_state = field, player
        self.label = player.label
        pos = self.pos2pix(player.pos)
        super().__init__(text=self.label, pos=pos, *args, **kwargs)
        self.width, self.height = player_width, player_width

    def on_touch_up(self, touch):
        if self.collide_point(touch.x, touch.y):
            self.player_state.pos = self.pix2pos()
            self.parent.update_img()
            return super().on_touch_up(touch)

    @property
    def scale(self):
        return max([self.field.h, self.field.w]) / field_height_m

    def pix2pos(self):
        offset = player_width / 2
        x, y = self.x, self.y
        x, y = y + offset, self.field.w - x - offset
        pos = np.array([x, y]) / self.scale
        pos[0] = field_width_m - pos[0]
        return pos

    def pos2pix(self, pos):
        x, y = pos
        x = field_width_m - x
        x, y = field_height_m - y, x
        return (np.array([x, y]) * self.scale - (player_width / 2)).astype(int).tolist()


class UltimateTacticsBoardApp(App):
    def build(self):
        return Builder.load_string(kv)


if __name__ == '__main__':
    UltimateTacticsBoardApp().run()
