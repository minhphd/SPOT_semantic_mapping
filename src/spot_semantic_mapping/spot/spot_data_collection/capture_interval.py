"""
capture_interval.py

Captures images and metadata from a Boston Dynamics Spot robot at regular intervals, 
saving depth/RGB images, camera intrinsics, and robot pose data for tasks like 
semantic mapping or trajectory logging.

Dependencies:
- bosdyn.client
- OpenCV (cv2)
- NumPy
- Python standard libraries (argparse, os, json, time, datetime)

Usage:
    python capture_interval.py --hostname <ROBOT_HOSTNAME> --outdir <OUTPUT_DIRECTORY> --interval 2.0

Functions:
- decode_depth(resp): Decodes depth images.
- decode_rgb(resp): Decodes RGB images.
- extract_intrinsics(resp): Extracts camera intrinsics.
- extract_odom_T_cam(resp): Extracts camera-to-odom transform.
- extract_odom_T_body(robot_state): Extracts body-to-odom transform.
- dataset_loop(image_client, robot_state_client, outdir, interval): Captures and saves data.
- main(): Handles argument parsing and initialization.
"""

import argparse
import os
import json
import time
from datetime import datetime

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_state import RobotStateClient

from bosdyn.client.frame_helpers import (
    get_a_tform_b,
    ODOM_FRAME_NAME,
    BODY_FRAME_NAME,
)

###############################################################################
# Camera Sources
###############################################################################

BODY_CAMERAS = {
    "frontleft": ["frontleft_depth_in_visual_frame", "frontleft_fisheye_image"],
    "frontright": ["frontright_depth_in_visual_frame", "frontright_fisheye_image"],
    "left": ["left_depth_in_visual_frame", "left_fisheye_image"],
    "right": ["right_depth_in_visual_frame", "right_fisheye_image"],
    "back": ["back_depth_in_visual_frame", "back_fisheye_image"],
}

HAND_CAMERAS = {
    "hand": [
        "hand_depth",
        "hand_color_in_hand_depth_frame",
        "hand_depth_in_hand_color_frame",
        "hand_color_image",
    ]
}


###############################################################################
# Decoding helpers
###############################################################################

def decode_depth(resp):
    """Decode Spot depth image from either RAW uint16 or PNG."""
    rows = resp.shot.image.rows
    cols = resp.shot.image.cols
    fmt = resp.shot.image.format

    if fmt == resp.shot.image.FORMAT_RAW:
        depth = np.frombuffer(resp.shot.image.data, dtype=np.uint16)
        return depth.reshape(rows, cols)

    if fmt == resp.shot.image.FORMAT_PNG:
        arr = np.frombuffer(resp.shot.image.data, dtype=np.uint8)
        depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return depth.astype(np.uint16)

    print("[WARN] Unknown depth format:", fmt)
    return None


def decode_rgb(resp):
    raw = np.frombuffer(resp.shot.image.data, dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


###############################################################################
# Intrinsics extraction
###############################################################################

def extract_intrinsics(resp):
    """
    Extract JSON-safe intrinsics from Spot ImageResponse.
    Handles SDK versions where skew/focal_length/principal_point are Vec2.
    """
    if not resp.source or not resp.source.pinhole:
        # Some depth-only streams may not expose pinhole intrinsics
        print(f"[WARN] No pinhole intrinsics for {resp.source.name}")
        return None

    intr = resp.source.pinhole.intrinsics

    def vec2_to_dict(v):
        if hasattr(v, "x") and hasattr(v, "y"):
            return {"x": float(v.x), "y": float(v.y)}
        # Older SDK: could be scalar
        try:
            return float(v)
        except Exception:
            return 0.0

    focal = vec2_to_dict(intr.focal_length)
    pp    = vec2_to_dict(intr.principal_point)
    skew  = vec2_to_dict(intr.skew)

    def safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    return {
        "focal_length": focal,
        "principal_point": pp,
        "skew": skew,
        "k1": safe_float(getattr(intr, "k1", 0.0)),
        "k2": safe_float(getattr(intr, "k2", 0.0)),
        "k3": safe_float(getattr(intr, "k3", 0.0)),
        "p1": safe_float(getattr(intr, "p1", 0.0)),
        "p2": safe_float(getattr(intr, "p2", 0.0)),
    }


###############################################################################
# Pose extraction (camera → ODOM)
###############################################################################

def extract_odom_T_cam(resp):
    """
    Get odom_T_cam for this image using transforms_snapshot in the response.
    This is the key transform for multi-timestamp mapping.
    """
    snapshot = resp.shot.transforms_snapshot
    cam_frame = resp.shot.frame_name_image_sensor  # camera optical frame

    # Use ODOM as the global frame for mapping (more stable than VISION).
    odom_T_cam = get_a_tform_b(snapshot, ODOM_FRAME_NAME, cam_frame)

    T = odom_T_cam
    return {
        "position": {"x": T.position.x, "y": T.position.y, "z": T.position.z},
        "rotation": {"x": T.rotation.x, "y": T.rotation.y, "z": T.rotation.z, "w": T.rotation.w},
    }


def extract_odom_T_body(robot_state):
    """
    Get odom_T_body from robot_state for trajectory logging.
    """
    snapshot = robot_state.kinematic_state.transforms_snapshot
    odom_T_body = get_a_tform_b(snapshot, ODOM_FRAME_NAME, BODY_FRAME_NAME)
    T = odom_T_body
    return {
        "position": {"x": T.position.x, "y": T.position.y, "z": T.position.z},
        "rotation": {"x": T.rotation.x, "y": T.rotation.y, "z": T.rotation.z, "w": T.rotation.w},
    }


###############################################################################
# Main logging loop
###############################################################################

def dataset_loop(image_client, robot_state_client, outdir, interval):

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "intrinsics"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "meta", "robot_pose"), exist_ok=True)

    print(f"[INFO] Logging dataset → {outdir}")
    print(f"[INFO] Interval = {interval}s")

    # Collect full list of camera sources
    all_sources = []
    for lst in BODY_CAMERAS.values():
        all_sources.extend(lst)
    for lst in HAND_CAMERAS.values():
        all_sources.extend(lst)

    # Save intrinsics once per camera source
    intrinsics_path = os.path.join(outdir, "intrinsics", "intrinsics.json")
    intrinsics_written = False
    intrinsics_dict = {}

    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Robot pose (odom_T_body) at this capture step
        try:
            state = robot_state_client.get_robot_state()
            odom_T_body = extract_odom_T_body(state)
        except Exception as e:
            print("[WARN] Failed to get robot state:", e)
            odom_T_body = None

        # Save concise robot pose (not full text dump)
        if odom_T_body is not None:
            pose_meta = {
                "timestamp": timestamp,
                "odom_T_body": odom_T_body,
            }
            pose_path = os.path.join(outdir, "meta", "robot_pose", f"{timestamp}.json")
            with open(pose_path, "w") as f:
                json.dump(pose_meta, f, indent=2)

        # Optional: still dump full robot_state text for debugging
        robot_state_txt_path = os.path.join(outdir, "meta", f"{timestamp}_robot_state.txt")
        with open(robot_state_txt_path, "w") as f:
            f.write(str(state))

        # Capture images from all sources
        try:
            responses = image_client.get_image_from_sources(all_sources)
        except Exception as e:
            print("[ERROR] Failed to capture images:", e)
            time.sleep(interval)
            continue

        print(f"[INFO] Captured {len(responses)} images @ {timestamp}")

        # Process each image
        for resp in responses:
            src = resp.source.name

            # -------------------------------------
            # Save intrinsics (once per camera)
            # -------------------------------------
            if not intrinsics_written:
                intr = extract_intrinsics(resp)
                if intr is not None:
                    intrinsics_dict[src] = intr

            # -------------------------------------
            # Save image data
            # -------------------------------------
            camdir = os.path.join(outdir, src)
            os.makedirs(camdir, exist_ok=True)
            img_path = os.path.join(camdir, f"{timestamp}.png")

            if resp.shot.image.pixel_format == resp.shot.image.PIXEL_FORMAT_DEPTH_U16:
                depth = decode_depth(resp)
                if depth is not None:
                    cv2.imwrite(img_path, depth)
            else:
                rgb = decode_rgb(resp)
                if rgb is not None:
                    cv2.imwrite(img_path, rgb)

            # -------------------------------------
            # Save per-image metadata
            # -------------------------------------
            odom_T_cam = extract_odom_T_cam(resp)

            acq_time = resp.shot.acquisition_time
            acquisition_time = {
                "sec": acq_time.seconds,
                "nsec": acq_time.nanos,
            }

            meta = {
                "timestamp": timestamp,
                "camera_source": src,
                "frame_name_image_sensor": resp.shot.frame_name_image_sensor,
                "world_frame": ODOM_FRAME_NAME,
                "odom_T_cam": odom_T_cam,
                "acquisition_time": acquisition_time,
            }

            meta_dir = os.path.join(outdir, "meta", src)
            os.makedirs(meta_dir, exist_ok=True)

            meta_path = os.path.join(meta_dir, f"{timestamp}.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        # After first cycle, write intrinsics
        if not intrinsics_written:
            with open(intrinsics_path, "w") as f:
                json.dump(intrinsics_dict, f, indent=2)
            intrinsics_written = True
            print("[INFO] Wrote intrinsics.")

        time.sleep(interval)


###############################################################################
# Entry point
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("--outdir", default="spot_dataset", help="Dataset directory")
    parser.add_argument("--interval", type=float, default=2.0, help="Capture interval (s)")
    options = parser.parse_args()

    # Clear out data folder at start
    if os.path.exists(options.outdir):
        # final query before clear
        response = input(f"Clear existing directory '{options.outdir}'? (y/n): ")
        if response.lower() != "y":
            print("[INFO] Aborted.")
            return
        import shutil
        shutil.rmtree(options.outdir)

    sdk = bosdyn.client.create_standard_sdk("spot_dataset_logger")
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    image_client = robot.ensure_client(ImageClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    print("[INFO] Connected to Spot.")

    try:
        dataset_loop(image_client, robot_state_client, options.outdir, options.interval)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")


if __name__ == "__main__":
    main()
