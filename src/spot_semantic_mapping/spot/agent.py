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
from spot_semantic_mapping.spot.decoding import decode_rgb, decode_depth


def authenticate_from_file(robot, cred_path: str):
    cred_path = Path(cred_path).expanduser()
    creds = json.loads(cred_path.read_text())
    robot.authenticate(creds["username"], creds["password"])


class SpotAgent:
    def __init__(self, hostname):
        self.cameras = {
            "hand": ["hand_color_image", "hand_depth_in_hand_color_frame"],
            "frontleft": ["frontleft_depth_in_visual_frame", "frontleft_fisheye_image"],
            "frontright": ["frontright_depth_in_visual_frame", "frontright_fisheye_image"],
            "left": ["left_depth_in_visual_frame", "left_fisheye_image"],
            "right": ["right_depth_in_visual_frame", "right_fisheye_image"],
            "back": ["back_depth_in_visual_frame", "back_fisheye_image"],
        }
        
        # Added pose to observation dictionary
        self.observation = {
            'battery': None,
            'cameras': {},
            'imu': {
                'angular_velocity': None,
                'linear_velocity': None
            },
            'pose': {
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'rotation': {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}
            }
        }
        
        self.sdk = bosdyn.client.create_standard_sdk("spot_agent")
        self.robot = self.sdk.create_robot(hostname)
        authenticate_from_file(self.robot, "./src/configs/spot_credentials.json")
        
        # clients
        self.image_client = self.robot.ensure_client('image')
        self.robot_state_client = self.robot.ensure_client('robot-state')
        self.local_grid_client = self.robot.ensure_client(LocalGridClient.default_service_name)


    async def get_local_grids_async(self, grid_types=None):
        """
        grid_types: list[str] like ["terrain", "obstacle_distance", ...]
        If None, you can first list what’s available.
        """
        if grid_types is None:
            # Discover what grids exist on the robot (names vary by config)
            grid_types = [g.name for g in self.local_grid_client.list_local_grids()]

        # Async call: get all requested grids in one request
        resp = await self.local_grid_client.get_local_grids_async(types=grid_types)

        # resp.local_grid_responses: list of grids
        grids_out = {}
        for g in resp.local_grid_responses:
            # Each g.local_grid has metadata + data. Data is typically a dense grid.
            # You’ll likely want to convert to numpy for downstream use.
            lg = g.local_grid
            grids_out[lg.local_grid_type_name] = {
                "frame_name": lg.frame_name,
                "cell_size": lg.extent.cell_size,
                "rows": lg.extent.num_cells_y,
                "cols": lg.extent.num_cells_x,
                "encoding": lg.encoding,  # tells how data is packed
                "data": lg.data,          # raw bytes / packed values
                "origin": {
                    "x": lg.extent.transform_snapshot.child_to_parent.x,
                    "y": lg.extent.transform_snapshot.child_to_parent.y,
                    "z": lg.extent.transform_snapshot.child_to_parent.z,
                },
            }

        return grids_out


    def _get_observation_sync(self):
        # 1) request images/state/grid concurrently using BD futures
        obs = {
            "battery": None,
            "imu": {"angular_velocity": None, "linear_velocity": None},
            "pose": {"position": None, "rotation": None},
            "local_grid": None,
            "cameras": {},   # NEW dict each call
        }

        flatten = []
        for cams in self.cameras.values():
            flatten.extend(cams)

        image_fut = self.image_client.get_image_from_sources_async(flatten)
        state_fut = self.robot_state_client.get_robot_state_async()
        grid_fut  = self.local_grid_client.get_local_grids_async(local_grid_type_names=["terrain", "obstacle_distance"])

        # 2) block until done
        camera_captures = image_fut.result()
        state = state_fut.result()
        grids = grid_fut.result()

        # obs = self.observation

        # Battery
        obs["battery"] = state.power_state.locomotion_charge_percentage.value

        # IMU / twist
        twist = state.kinematic_state.velocity_of_body_in_vision
        obs["imu"]["angular_velocity"] = {"x": twist.angular.x, "y": twist.angular.y, "z": twist.angular.z}
        obs["imu"]["linear_velocity"]  = {"x": twist.linear.x,  "y": twist.linear.y,  "z": twist.linear.z}

        # Pose
        vision_tform_body = get_a_tform_b(state.kinematic_state.transforms_snapshot, "vision", "body")
        obs["pose"]["position"] = {"x": vision_tform_body.x, "y": vision_tform_body.y, "z": vision_tform_body.z}
        obs["pose"]["rotation"] = {
            "w": vision_tform_body.rot.w,
            "x": vision_tform_body.rot.x,
            "y": vision_tform_body.rot.y,
            "z": vision_tform_body.rot.z,
        }

        # Local grid (raw response for now)
        obs["local_grid"] = grids

        # Cameras
        for camera in camera_captures:
            src = camera.source.name
            location = src.split("_")[0]
            obs["cameras"].setdefault(location, {})
            if camera.shot.image.pixel_format == camera.shot.image.PIXEL_FORMAT_DEPTH_U16:
                obs["cameras"][location]["depth"] = decode_depth(camera)
            else:
                obs["cameras"][location]["color"] = decode_rgb(camera)

        return obs

    async def get_observation_async(self):
        # Run the blocking SDK calls in a worker thread
        return await asyncio.to_thread(self._get_observation_sync)