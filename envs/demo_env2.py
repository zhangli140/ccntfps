# coding: utf-8

import math, time, random
import numpy as np
from gym import spaces
from .import FPS_env as fps

from ..utils import *

class DemoEnv2(fps.FPSEnv):
    '''
    TODO
    '''
    def __init__(self,):
        super(DemoEnv, self).__init__()

    def _step(self, action):
        return 0,0,0,''

    def _reset(self, ):
        return 0
