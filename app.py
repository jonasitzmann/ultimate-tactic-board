from kivy.config import Config
from dataclasses import dataclass
import mode
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
from collections import deque
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
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.relativelayout import RelativeLayout
from kivy.core.window import Window
from cfg import field_height_m, field_width_m
from state import State
import manim_animations
from contextlib import contextmanager

Window.size, disc_width = (1800, 800), 40

@contextmanager
def cd(path):
    old_path = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(old_path)



def get_play_dir():
    os.makedirs(cfg.plays_dir, exist_ok=True)
    dirs = [d.split('/')[-2] for d in glob(f'{cfg.plays_dir}/*/')]
    max_dir = max([int(d) for d in dirs if d.isnumeric()] or [0]) if dirs else 0
    play_number = max_dir + 1
    new_dir = f'{cfg.plays_dir}/{play_number}'
    os.makedirs(new_dir)
    return new_dir, play_number


@dataclass
class Configuration:
    state = State()
    frame_number = 1
    annotations = []



class Field(RelativeLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.play_dir, self.play_number = get_play_dir()
        self.annotations = []
        self.configuration = Configuration()
        self.previous_state = None
        self.state_img = manim_animations.StateImg()
        self.size_hint = (None, None)
        self.frame_number = 1
        self.current_player_role = None
        self.current_player = None
        self.anglemode = False
        self.play_name = None
        hist_size = 50
        self.undo_cmds = deque(maxlen=hist_size)
        self.redo_cmds = deque(maxlen=hist_size)
        with self.canvas:
            self.image = kiImage(size_hint=(None, None))
        self.add_widget(self.update_img())
        self.mode_text = 'view'
        self.set_mode()

    @property
    def state(self):
        return self.configuration.state

    @state.setter
    def state(self, value):
        self.configuration.state = value

    @property
    def w(self):
        return self.image.width

    @property
    def h(self):
        return self.image.height

    @property
    def num_frames(self):
        return len(glob(f'{self.play_dir}/*.yaml'))

    def execute_cmd(self, cmd):
        self.undo_cmds.append(cmd)
        self.redo_cmds.clear()
        self.do_and_reload(cmd.execute)

    def undo(self):
        if self.undo_cmds:
            cmd = self.undo_cmds.pop()
            self.redo_cmds.append(cmd)
            self.do_and_reload(cmd.undo)

    def redo(self):
        if self.redo_cmds:
            cmd = self.redo_cmds.pop()
            self.undo_cmds.append(cmd)
            self.do_and_reload(cmd.execute)

    def do_and_reload(self, func):
        func()
        self.update_img()


    def execute_text_command(self, command):
        command = 'self.state.' + command
        try:
            exec(command)
            self.update_img()
            self.set_mode()
        except Exception as e:
            print(e)
        finally:
            self.text_input.text = ''

    def set_mode(self, mode_text=None):
        if not mode_text:
            mode_text = self.mode_text
        self.mode_text = mode_text
        if mode_text == 'add o':
            self.mode = mode.AddPlayerMode(self, 'o')
        elif mode_text == 'add d':
            self.mode = mode.AddPlayerMode(self, 'd')
        elif mode_text == 'move':
            self.mode = mode.EditPoseMode(self)
        elif mode_text == 'select':
            self.mode = mode.SelectMode(self)
        elif mode_text == 'view':
            self.mode = mode.ViewMode(self)

    def prepare_animation_script(self):
        with open(cfg.template_play_file, 'r') as f:
            template = f.read()
        if self.play_name is None:
            self.play_name = f'play{self.play_number}'
        template = template.replace('TemplateScene', self.play_name.capitalize())
        template = template.replace('play_name', self.play_name)
        indentation = ' ' * 8
        transitions = '\n'.join([f'{indentation}f.transition(s[{i}], run_time=2)' for i in range(1, self.num_frames)])
        template = template.replace(f'{indentation}# state transitions', transitions)
        save_path = f'{self.play_dir}/{self.play_name}.py'
        with open(save_path, 'w') as f:
            f.write(template)

    def execute_animation_script(self):
        sys.path.append(self.play_dir)
        script = importlib.import_module(f'{self.play_name}')
        with cd(self.play_dir):
            script.render_scene()

    def render(self):
        self.execute_animation_script()

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.mode.on_touch_down(touch)

    def on_touch_up(self, touch):
        self.mode.on_touch_up(touch)

    def save_state(self):
        self.state.save(f'{self.play_dir}/{self.frame_number}.yaml')
        self.prepare_animation_script()
        self.load_frame(self.frame_number + 1)

    def load_state(self, frame_number):
        filename = f'{self.play_dir}/{frame_number}.yaml'
        if os.path.exists(filename):
            return State.load(filename)
        return None

    def load_frame(self, frame_number):
        self.frame_number = max(frame_number, 1)
        self.frame_number_label.text = str(self.frame_number)
        new_state = self.load_state(self.frame_number)
        self.state = self.state if new_state is None else new_state
        self.previous_state = self.load_state(self.frame_number - 1)
        self.set_mode()

    def reset(self):
        self.clear_widgets()
        self.add_widget(self.update_img())

    @property
    def scale(self):
        return max([self.h, self.w]) / field_height_m

    def update_img(self):
        img_data = self.state_img.get_img(self.state, self.configuration.annotations)
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
        except Exception as e:
            print(f'error\n{e}')

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

    def get_previous_player(self, player):
        if self.previous_state is None:
            return None
        return self.previous_state.get_player(player)


class UltimateTacticsBoardApp(App):
    def build(self):
        with open('kv_app.kv', 'r') as f:
            kv = f.read()
        return Builder.load_string(kv)


if __name__ == '__main__':
    UltimateTacticsBoardApp().run()