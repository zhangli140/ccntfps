from gym.envs.registration import register
from . import client
from .envs.demo_env import DemoEnv
from .envs.demo_env2 import DemoEnv2
from .envs.demo_env3 import DemoEnv3

register(
    id='FPSDemo-v0',
    entry_point='gym_FPS:DemoEnv',)
register(
    id='FPSDemo-v2',
    entry_point='gym_FPS:DemoEnv2',)
register(
    id='FPSDemo-v3',
    entry_point='gym_FPS:DemoEnv3',)