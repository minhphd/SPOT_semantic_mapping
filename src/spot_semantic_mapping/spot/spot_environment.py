import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_state import RobotStateClient
import asyncio
import cv2
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.client.frame_helpers import (
    get_a_tform_b,
    ODOM_FRAME_NAME,
    BODY_FRAME_NAME,
)
import re
import numpy as np
import json
from pathlib import Path


class SpotEnv:
    def __init__(self):
        self.logger = None
        self.reward = 0
        self.graph = None
        self.agent = None
        self.action_space = None
        self.observation_space = None
        self.np_random = None
        
        
    def reset(self):
        pass
    
    def step(self):
        pass
    
    def close(self):
        pass

    