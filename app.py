from manim_animations import Field as MField
from functools import partial
from kivy.config import Config
from dataclasses import dataclass
from kivy.properties import ObjectProperty
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup

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



class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

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
        self.mode = None
        self.play_dir, self.play_number = None, None
        self.annotations = []
        self.configuration = Configuration()
        self.state_img = manim_animations.StateImg()
        self.size_hint = (None, None)
        self.frame_number = 1
        self.current_player = None
        self.anglemode = False
        self.play_name = None
        self.grid_size = 0.5
        hist_size = 50
        self.undo_cmds = deque(maxlen=hist_size)
        self.redo_cmds = deque(maxlen=hist_size)
        with self.canvas:
            self.image = kiImage(size_hint=(None, None))
        self.add_widget(self.update_img())
        self.mode_text = 'view'
        self.set_mode()
        self.filename = 'not saved'
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_key_down)
        self._keyboard.bind(on_key_up=self._on_key_up)
        self.children[0].bind(on_touch_down=self.on_touch_down)
        self.pressed_keys = set()

    def on_press(self, *args, **kwargs):
        print(*args)

    @property
    def previous_state(self):
        return self.load_state(self.frame_number - 1)

    @property
    def ctrl_pressed(self):
        return self.is_pressed('rctrl', 'lctrl')


    def is_pressed(self, *keys):
        keycodes = [self._keyboard.keycodes.get(k, None) for k in keys]
        return any(k in self.pressed_keys for k in keycodes)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_key_down)
        self._keyboard.unbind(on_key_up=self._on_key_up)
        self._keyboard = None

    def _on_key_down(self, keyboard, keycode, text, modifiers):
        if keycode not in self.pressed_keys:
            self.pressed_keys.add(keycode[0])

    def _on_key_up(self, keyboard, keycode):
        if keycode[0] in self.pressed_keys:
            self.pressed_keys.remove(keycode[0])

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
        self.mode.reload()


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
        if mode_text == 'add':
            self.set_mode_(mode.AddPlayerMode(self))
        elif mode_text == 'move':
            self.set_mode_(mode.EditPoseMode(self))
        elif mode_text == 'select':
            self.set_mode_(mode.SelectMode(self))
        elif mode_text == 'view':
            self.set_mode_(mode.ViewMode(self))
        elif mode_text == 'hex':
            self.set_mode_(mode.SetupHexMode(self))

    def set_mode_(self, mode_):
        if self.mode is not None:
            self.mode.__del__()
        self.mode = mode_

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
            if touch.button == 'middle' or self.is_pressed('shift'):
                pos = self.pix2pos(*touch.pos)
                disc_pos = pos
                self.state.disc = disc_pos
                self.update_img()
            elif touch.button == 'mouse4':  # undo
                self.undo()
            elif touch.button == 'mouse5':  # redo
                self.redo()
            else:
                self.mode.on_touch_down(touch)

    def on_touch_up(self, touch):
        self.mode.on_touch_up(touch)

    def on_touch_move(self, touch):
        self.mode.on_touch_move(touch)

    def save_state(self):
        if self.play_dir is None:
            self.play_dir, self.play_number = get_play_dir()
        self.state.save(f'{self.play_dir}/{self.frame_number}.yaml')
        self.prepare_animation_script()
        self.load_frame(self.frame_number + 1)
        self.filename = 'not saved'

    def dismiss_popup(self):
        self._popup.dismiss()

    def load_dialog(self):
        content = LoadDialog(load=self.load_file, cancel=self.dismiss_popup)
        PATH = "."
        content.ids.filechooser.path = PATH
        self._popup = Popup(title="Load file", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def load_template_dialog(self):
        content = LoadDialog(load=self.load_template_file, cancel=self.dismiss_popup)
        PATH = "."
        content.ids.filechooser.path = PATH
        self._popup = Popup(title="Load file", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def load_template_file(self, filename):
        self.state = State.load(filename)
        self.update_img()
        self.dismiss_popup()

    def load_file(self, filename):
        self.play_dir, filename = os.path.split(filename)
        self.play_name = os.path.basename(self.play_dir)
        self.frame_number = int(os.path.splitext(filename)[0])
        self.load_frame(self.frame_number)
        self.dismiss_popup()

    def update_description(self):
        text = f'play: {self.play_name}\nframe: {self.frame_number}'
        if self.filename == 'not saved':
            text += '\n(unsaved)'
        self.frame_number_label.text = text


    def load_state(self, frame_number):
        filename = f'{self.play_dir}/{frame_number}.yaml'
        if os.path.exists(filename):
            return State.load(filename)

    def load_frame(self, frame_number):
        self.frame_number = max(frame_number, 1)
        new_state = self.load_state(self.frame_number)
        if new_state is not None:
            self.state = new_state
            self.filename = str(self.frame_number)
        else:
            self.filename = 'not saved'
        self.state = self.state if new_state is None else new_state
        self.update_description()
        self.set_mode()

    def reset(self):
        self.clear_widgets()
        self.add_widget(self.update_img())

    @property
    def scale(self):
        return max([self.h, self.w]) / field_height_m

    def update_img(self):
        annotations = self.configuration.annotations
        past_arrows = partial(MField.arrows, state=self.load_state(self.frame_number - 1), next_state=self.state)
        future_arrows = partial(MField.arrows, state=self.state, next_state=self.load_state(self.frame_number + 1))
        annotations += [past_arrows, future_arrows]
        img_data = self.state_img.get_img(self.state, self.configuration.annotations)
        self.configuration.annotations = [a for a in annotations if a not in [past_arrows, future_arrows]]
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

    def pix2pos(self, x, y, offset=0):
        x, y = y + offset, x + offset
        pos = np.array([x, y]) / self.scale
        pos[0] = cfg.field_width_m - pos[0]
        pos = np.round(pos / self.grid_size) * self.grid_size
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