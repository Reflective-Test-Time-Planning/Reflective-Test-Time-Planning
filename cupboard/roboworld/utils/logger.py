import time
import cloudpickle as pickle
import tempfile
import os
import sys
import random
from copy import deepcopy
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import config_dict
import uuid
from socket import gethostname
import wandb
import numpy as np
import absl.flags
from absl import logging
import pprint


class Timer(object):

    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time

class WandBLogger(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = True
        config.prefix = ''
        config.project = 'RoboVLM'
        config.group = '1'
        config.output_dir = '/tmp/robo_vlm'
        config.random_delay = 0.0
        config.save_video = False
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant):
        self.config = self.get_default_config(config)

        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != '':
            self.config.project = '{}--{}'.format(self.config.prefix, self.config.project)

        if self.config.output_dir == '':
            self.config.output_dir = tempfile.mkdtemp()
        else:
            self.config.output_dir = os.path.join(self.config.output_dir, self.config.experiment_id)
            os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = deepcopy(variant)

        if 'hostname' not in self._variant:
            self._variant['hostname'] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        self.run = wandb.init(
            reinit=False,
            config=self._variant,
            project=self.config.project,
            group=self.config.group,
            dir=self.config.output_dir,
            name=self.config.experiment_id,
            anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode='online' if self.config.online else 'offline',
        )

        self._save_video = self.config.save_video

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def log_video(self, data_or_path, caption=None, fps=30, format="mp4", label='video'):
        self.log({label: wandb.Video(data_or_path, caption=caption, fps=fps, format=format)})

    def save_pickle(self, obj, filename):
        filepath = os.path.join(self.config.output_dir, filename)
        succ = False
        cnt = 0
        while not succ:
            cnt += 1
            try:
                with open(filepath, 'wb') as fout:
                    pickle.dump(obj, fout, protocol=4)
            except IOError:
                print(f"pickle dump attempt {cnt} failed.")
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            else:
                succ = True

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir


def prefix_metrics(metrics, prefix):
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
    }


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, 'automatically defined flag')
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, 'automatically defined flag')
        else:
            raise ValueError(f'Incorrect value type for {key}')
    return kwargs


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
