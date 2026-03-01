from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterator, Union
import json
from datetime import datetime
import pandas as pd
import numpy as np
import cv2


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class PoseSE3:
    """SE(3) pose as translation + quaternion (x,y,z,w)."""
    position: np.ndarray  # (3,) float32
    quat_xyzw: np.ndarray # (4,) float32

    def as_matrix(self) -> np.ndarray:
        """4x4 homogeneous matrix."""
        x, y, z, w = self.quat_xyzw.tolist()
        # quaternion -> rotation matrix
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        R = np.array([
            [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
            [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
        ], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3,  3] = self.position
        return T


@dataclass
class CameraIntrinsics:
    """Spot pinhole intrinsics in JSON-safe form."""
    fx: float
    fy: float
    cx: float
    cy: float
    skew_x: float = 0.0
    skew_y: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0


@dataclass
class SensorFrame:
    """
    One camera stream at one timestamp.

    image: loaded image array (RGB uint8 HxWx3 or depth uint16 HxW)
    meta:  meta JSON dict for this camera at this timestamp
    intr:  intrinsics for this camera (may be None if missing)
    odom_T_cam: SE(3) Pose of camera in ODOM frame (may be None)
    """
    source: str
    timestamp: str
    image: Optional[np.ndarray]
    meta: Optional[Dict[str, Any]]
    intr: Optional[CameraIntrinsics]
    odom_T_cam: Optional[PoseSE3]


@dataclass
class SpotSample:
    """
    One synchronized capture step (one timestamp), with:
      - robot pose (odom_T_body)
      - dictionary of camera frames
    """
    timestamp: str
    odom_T_body: Optional[PoseSE3]
    cameras: Dict[str, SensorFrame]


# -----------------------------
# Dataloader
# -----------------------------
class SpotDataset:
    """
    Dataloader over a folder produced by capture_interval.py.

    Layout expected:
      outdir/
        intrinsics/intrinsics.json
        meta/
          robot_pose/<timestamp>.json
          <camera_source>/<timestamp>.json
          <timestamp>_robot_state.txt  (ignored)
        <camera_source>/<timestamp>.png

    Usage:
      ds = SpotDataset("spot_dataset", load_images=True)
      sample = ds[0]
      frame = sample.cameras["frontleft_fisheye_image"]
      rgb = frame.image
      T = frame.odom_T_cam.as_matrix()
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        load_images: bool = True,
        image_mode: str = "auto",   # "auto" | "rgb" | "depth"
        allowed_sources: Optional[List[str]] = None,
        strict: bool = False,       # if True, missing files raise
    ):
        self.root = Path(root_dir)
        self.load_images = load_images
        self.image_mode = image_mode
        self.allowed_sources = set(allowed_sources) if allowed_sources else None
        self.strict = strict

        self.meta_dir = self.root / "meta"
        self.intr_path = self.root / "intrinsics" / "intrinsics.json"

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        if not self.meta_dir.exists():
            raise FileNotFoundError(f"Missing meta/ directory: {self.meta_dir}")

        self.intrinsics_raw: Dict[str, Any] = self._load_json(self.intr_path) if self.intr_path.exists() else {}
        self.intrinsics: Dict[str, CameraIntrinsics] = self._parse_intrinsics(self.intrinsics_raw)

        self.camera_sources: List[str] = self._discover_camera_sources()
        self.timestamps: List[str] = self._discover_timestamps()

    # --------- public API ---------
    def __len__(self) -> int:
        return len(self.timestamps)

    def __iter__(self) -> Iterator[SpotSample]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, t: int) -> SpotSample:
        if t < 0 or t >= len(self.timestamps):
            raise IndexError(f"t={t} out of range [0, {len(self.timestamps)-1}]")
        ts = self.timestamps[t]
        return self.get_by_timestamp(ts)

    def get_by_timestamp(self, ts: str) -> SpotSample:
        # robot pose
        odom_T_body = None
        pose_path = self.meta_dir / "robot_pose" / f"{ts}.json"
        if pose_path.exists():
            pose_meta = self._load_json(pose_path)
            odom_T_body = self._pose_from_meta(pose_meta.get("odom_T_body"))
        elif self.strict:
            raise FileNotFoundError(f"Missing robot pose file: {pose_path}")

        cameras: Dict[str, SensorFrame] = {}

        for src in self.camera_sources:
            if self.allowed_sources is not None and src not in self.allowed_sources:
                continue

            img_path = self.root / src / f"{ts}.png"
            meta_path = self.meta_dir / src / f"{ts}.json"

            if not img_path.exists() and not meta_path.exists():
                continue

            meta = self._load_json(meta_path) if meta_path.exists() else None
            intr = self.intrinsics.get(src)

            odom_T_cam = None
            if meta is not None:
                odom_T_cam = self._pose_from_meta((meta.get("odom_T_cam") or {}).get("position") and meta.get("odom_T_cam"))

            img = None
            if self.load_images:
                if img_path.exists():
                    img = self._load_image(img_path, src=src)
                elif self.strict:
                    raise FileNotFoundError(f"Missing image: {img_path}")

            cameras[src] = SensorFrame(
                source=src,
                timestamp=ts,
                image=img,
                meta=meta,
                intr=intr,
                odom_T_cam=odom_T_cam,
            )

        return SpotSample(timestamp=ts, odom_T_body=odom_T_body, cameras=cameras)

    # --------- helpers ---------
    def _discover_camera_sources(self) -> List[str]:
        # Prefer meta subfolders (more reliable)
        sources = []
        for p in self.meta_dir.iterdir():
            if p.is_dir() and p.name != "robot_pose":
                sources.append(p.name)

        # Also include top-level directories that are not meta/intrinsics
        for p in self.root.iterdir():
            if p.is_dir() and p.name not in {"meta", "intrinsics"}:
                sources.append(p.name)

        sources = sorted(set(sources))
        return sources

    def _discover_timestamps(self) -> List[str]:
        ts_set = set()

        # robot_pose timestamps
        pose_dir = self.meta_dir / "robot_pose"
        if pose_dir.exists():
            for j in pose_dir.glob("*.json"):
                ts_set.add(j.stem)

        # per-camera timestamps
        for src in self.camera_sources:
            src_meta = self.meta_dir / src
            if not src_meta.exists():
                continue
            for j in src_meta.glob("*.json"):
                ts_set.add(j.stem)

        ts = sorted(ts_set)
        if self.strict and len(ts) == 0:
            raise RuntimeError(f"No timestamps found under: {self.meta_dir}")
        return ts

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _parse_intrinsics(raw: Dict[str, Any]) -> Dict[str, CameraIntrinsics]:
        out: Dict[str, CameraIntrinsics] = {}

        def v2(d, kx="x", ky="y") -> Tuple[float, float]:
            if isinstance(d, dict):
                return float(d.get(kx, 0.0)), float(d.get(ky, 0.0))
            # fallback: scalar
            return float(d), float(d)

        for src, intr in raw.items():
            try:
                fx, fy = v2(intr.get("focal_length", {"x": 0.0, "y": 0.0}))
                cx, cy = v2(intr.get("principal_point", {"x": 0.0, "y": 0.0}))
                sx, sy = v2(intr.get("skew", {"x": 0.0, "y": 0.0}))
                out[src] = CameraIntrinsics(
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    skew_x=sx, skew_y=sy,
                    k1=float(intr.get("k1", 0.0)),
                    k2=float(intr.get("k2", 0.0)),
                    k3=float(intr.get("k3", 0.0)),
                    p1=float(intr.get("p1", 0.0)),
                    p2=float(intr.get("p2", 0.0)),
                )
            except Exception:
                # ignore malformed intrinsics
                continue
        return out

    @staticmethod
    def _pose_from_meta(pose: Any) -> Optional[PoseSE3]:
        """
        pose can be:
          {"position":{x,y,z},"rotation":{x,y,z,w}}
        """
        if not isinstance(pose, dict):
            return None
        pos = pose.get("position")
        rot = pose.get("rotation")
        if not (isinstance(pos, dict) and isinstance(rot, dict)):
            return None
        p = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float32)
        q = np.array([rot["x"], rot["y"], rot["z"], rot["w"]], dtype=np.float32)
        return PoseSE3(position=p, quat_xyzw=q)

    def _load_image(self, path: Path, src: str) -> np.ndarray:
        """
        Auto-detect depth vs rgb if image_mode='auto':
          - heuristic: if "depth" in source name -> depth
          - else load as BGR -> convert to RGB
        Depth is stored as uint16 PNG (Spot depth U16).
        """
        mode = self.image_mode
        
        # rotate 90 degrees if camera is in {'left_fisheye_image', 'frontright_fisheye_image', 'frontleft_fisheye_image'}
        rotation = False
        flip = False
        if src in {"frontright_fisheye_image", "frontleft_fisheye_image"}:
            rotation = True
        if src == "right_fisheye_image":
            flip = True
        
        if mode == "auto":
            mode = "depth" if "depth" in src.lower() else "rgb"

        if mode == "depth":
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                if self.strict:
                    raise FileNotFoundError(f"Failed to read: {path}")
                return None
            # ensure uint16
            if rotation:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            if flip:
                img = cv2.flip(img, 0)
            return img.astype(np.uint16)

        # rgb
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            if self.strict:
                raise FileNotFoundError(f"Failed to read: {path}")
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rotation:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        if flip:
            rgb = cv2.flip(rgb, 0)
        return rgb

    def generate_strayscanner_dir(self, camera='hand_color', clear_existing: bool = True) -> None:
        """
        Generate a StrayScanner-compatible directory structure from the dataset.

        StrayScanner expects:
        root_dir/
            camera_matrix.csv
            imu.csv (timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_y, alpha_z)
            odometry.csv (headers: timestamp, frame, x, y, z, qx, qy, qz, qw)
            confidence/
            <timestamp>.json
            rgb_frames/
            <timestamp>.png
            depth/
            <timestamp>.png

        Notes:
        - This dataset format does not include IMU streams, so imu.csv is emitted as a header-only placeholder.
        - We choose one RGB stream and one depth stream heuristically.
        - We export odometry from meta/robot_pose/<timestamp>.json (odom_T_body).
        """
        import csv
        
        out_root = self.root.parent / "strayscanner"
        
        if clear_existing and out_root.exists():
            import shutil
            shutil.rmtree(out_root)
        
        rgb_dir = out_root / "rgb_frames"
        depth_dir = out_root / "depth"
        conf_dir = out_root / "confidence"
        out_root.mkdir(parents=True, exist_ok=True)
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        conf_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Choose representative sources
        # -----------------------------
        sources = list(self.camera_sources)

        def is_depth_src(s: str) -> bool:
            return "depth" in s.lower()

        def is_rgb_src(s: str) -> bool:
            # Most Spot RGB sources end with "_image" and are not depth
            sl = s.lower()
            return (("image" in sl) or ("fisheye" in sl) or ("color" in sl) or ("rgb" in sl)) and not is_depth_src(s)

        # Prefer an RGB source that has intrinsics
        rgb_candidates = [s for s in sources if is_rgb_src(s)]
        rgb_candidates_intr = [s for s in rgb_candidates if s in self.intrinsics]
        # rgb_source = f'hand_color_image'
        rgb_source = 'left_fisheye_image'
        # rgb_source = (rgb_candidates_intr[0] if rgb_candidates_intr else (rgb_candidates[0] if rgb_candidates else None))

        depth_candidates = [s for s in sources if is_depth_src(s)]
        depth_candidates_intr = [s for s in depth_candidates if s in self.intrinsics]
        depth_source = 'left_depth_in_visual_frame'
        # depth_source = (depth_candidates_intr[0] if depth_candidates_intr else (depth_candidates[0] if depth_candidates else None))

        if rgb_source is None and depth_source is None:
            if self.strict:
                raise RuntimeError("No camera sources found to export (neither RGB nor depth).")
            return

        # -----------------------------
        # Write camera_matrix.csv (K)
        # -----------------------------
        cam_intr = None
        if rgb_source is not None:
            cam_intr = self.intrinsics.get(rgb_source)
        if cam_intr is None and depth_source is not None:
            cam_intr = self.intrinsics.get(depth_source)

        cam_K_path = out_root / "camera_matrix.csv"
        if cam_intr is not None:
            K = np.array(
                [
                    [cam_intr.fx, cam_intr.skew_x, cam_intr.cx],
                    [cam_intr.skew_y, cam_intr.fy, cam_intr.cy],
                    [0.0,         0.0,           1.0],
                ],
                dtype=np.float32,
            )
            # Write as 3 rows, 3 columns
            with open(cam_K_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for r in range(3):
                    w.writerow([f"{K[r, c]:.9f}" for c in range(3)])
        else:
            # Still emit something predictable
            with open(cam_K_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["1.0", "0.0", "0.0"])
                w.writerow(["0.0", "1.0", "0.0"])
                w.writerow(["0.0", "0.0", "1.0"])

        # -----------------------------
        # Write imu.csv placeholder
        # -----------------------------
        imu_path = out_root / "imu.csv"
        with open(imu_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "a_x", "a_y", "a_z", "alpha_x", "alpha_y", "alpha_z"])
            # No rows (IMU not captured in this dataset layout)

        # -----------------------------
        # Write odometry.csv
        # -----------------------------
        odo_path = out_root / "odometry.csv"
        idx = 0
        format_str = "%Y%m%d_%H%M%S"
        with open(odo_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw"])
            idx = 0
            # start_time = None
            for ts in self.timestamps:
                # if start_time is None:
                #     start_time = datetime.strptime(ts, format_str).timestamp()
                pose_path = self.meta_dir / "robot_pose" / f"{ts}.json"
                if not pose_path.exists():
                    if self.strict:
                        raise FileNotFoundError(f"Missing robot pose file: {pose_path}")
                    continue

                pose_meta = self._load_json(pose_path)
                odom_T_body = self._pose_from_meta(pose_meta.get("odom_T_body"))
                if odom_T_body is None:
                    if self.strict:
                        raise ValueError(f"Malformed odom_T_body in: {pose_path}")
                    continue
        
                file_name = f"{idx:06d}.png"
                # Load meta if present (for confidence JSON)
                rgb_meta = None
                depth_meta = None

                # RGB
                rgb_written = False
                if rgb_source is not None:
                    rgb_img_path = self.root / rgb_source / f"{ts}.png"
                    rgb_meta_path = self.meta_dir / rgb_source / f"{ts}.json"
                    if rgb_meta_path.exists():
                        rgb_meta = self._load_json(rgb_meta_path)

                    if rgb_img_path.exists():
                        self.image_mode = "rgb"
                        rgb = self._load_image(rgb_img_path, src=rgb_source)
                        if rgb is not None:
                            # cv2 wants BGR
                            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            cv2.imwrite(str(rgb_dir / file_name), bgr)
                            rgb_written = True
                    elif self.strict and (self.meta_dir / rgb_source).exists():
                        raise FileNotFoundError(f"Missing RGB image: {rgb_img_path}")

                # Depth
                depth_written = False
                if depth_source is not None:
                    d_img_path = self.root / depth_source / f"{ts}.png"
                    d_meta_path = self.meta_dir / depth_source / f"{ts}.json"
                    if d_meta_path.exists():
                        depth_meta = self._load_json(d_meta_path)

                    if d_img_path.exists():
                        self.image_mode = "depth"
                        dep = self._load_image(d_img_path, src=depth_source)
                        if dep is not None:
                            # Ensure uint16 png
                            dep_u16 = dep.astype(np.uint16)
                            cv2.imwrite(str(depth_dir / file_name), dep_u16)
                            depth_written = True
                    elif self.strict and (self.meta_dir / depth_source).exists():
                        raise FileNotFoundError(f"Missing depth image: {d_img_path}")

                # Confidence values (0, 1, 2)
                if depth_written and rgb_written:
                    conf = np.ones((dep.shape[0], dep.shape[1]), dtype=np.uint8)
                    x, y, z = odom_T_body.position.tolist()
                    qx, qy, qz, qw = odom_T_body.quat_xyzw.tolist()
                    ts_float = datetime.strptime(ts, format_str).timestamp()
                    w.writerow([ts_float, f"{idx:06d}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{qx:.8f}", f"{qy:.8f}", f"{qz:.8f}", f"{qw:.8f}"])
                    cv2.imwrite(str(conf_dir / file_name), conf)
                    idx += 1
