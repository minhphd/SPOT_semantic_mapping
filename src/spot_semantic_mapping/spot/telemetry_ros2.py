import threading
import asyncio
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image, BatteryState
from geometry_msgs.msg import PoseStamped, TwistStamped
from cv_bridge import CvBridge
from agent import SpotAgent
from sensor_msgs.msg import Image as RosImage

def np_to_ros_image(img_np, frame_id, stamp, encoding="bgr8"):
    msg = RosImage()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(img_np.shape[0])
    msg.width = int(img_np.shape[1])

    if encoding == "bgr8":
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = int(msg.width * 3)
        msg.data = img_np.astype(np.uint8).tobytes()
    elif encoding == "mono8":
        msg.encoding = "mono8"
        msg.is_bigendian = False
        msg.step = int(msg.width)
        msg.data = img_np.astype(np.uint8).tobytes()
    elif encoding == "16UC1":
        msg.encoding = "16UC1"
        msg.is_bigendian = False
        msg.step = int(msg.width * 2)
        msg.data = img_np.astype(np.uint16).tobytes()
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    return msg

def now_header(node: Node, frame_id: str) -> Header:
    h = Header()
    h.stamp = node.get_clock().now().to_msg()
    h.frame_id = frame_id
    return h


class SpotObservationPublisher(Node):
    def __init__(self, spot_agent, publish_rate_hz=5.0):
        super().__init__("spot_observation_publisher")
        self.agent = spot_agent
        self.bridge = CvBridge()

        # --- pubs ---
        self.pub_pose = self.create_publisher(PoseStamped, "/spot/pose", 10)
        self.pub_twist = self.create_publisher(TwistStamped, "/spot/twist", 10)
        self.pub_batt = self.create_publisher(BatteryState, "/spot/battery", 10)
        self.pub_img = {}   # topic -> publisher

        self.period = 1.0 / float(publish_rate_hz)

        # ---- asyncio in background thread ----
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        # state: last obs + in-flight future
        self._last_obs = None
        self._future = None

        # Kick off first request immediately
        self._schedule_fetch()

        self.timer = self.create_timer(self.period, self._tick)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def destroy_node(self):
        # stop asyncio loop cleanly
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        super().destroy_node()

    def _get_img_pub(self, topic: str):
        if topic not in self.pub_img:
            self.pub_img[topic] = self.create_publisher(Image, topic, 10)
        return self.pub_img[topic]

    def _schedule_fetch(self):
        # only schedule if nothing in flight
        if self._future is None or self._future.done():
            self._future = asyncio.run_coroutine_threadsafe(
                self.agent.get_observation_async(),
                self._loop
            )

    def _tick(self):
        # 1) if a fetch finished, grab it (non-blocking)
        if self._future is not None and self._future.done():
            try:
                self._last_obs = self._future.result()
            except Exception as e:
                self.get_logger().error(f"Failed to get observation: {e}")
                self._last_obs = None
            finally:
                self._future = None  # allow reschedule

        # 2) schedule next fetch if needed
        self._schedule_fetch()

        # 3) publish last observation if we have one
        obs = self._last_obs
        if obs is None:
            return

        # ----------------
        # Battery
        # ----------------
        b = BatteryState()
        b.header = now_header(self, "spot_body")
        batt_pct = obs.get("battery", 0.0)
        b.percentage = float(batt_pct) / 100.0
        self.pub_batt.publish(b)

        # ----------------
        # Twist
        # ----------------
        twist = TwistStamped()
        twist.header = now_header(self, "spot_body")
        av = obs["imu"]["angular_velocity"]
        lv = obs["imu"]["linear_velocity"]
        twist.twist.angular.x = float(av["x"])
        twist.twist.angular.y = float(av["y"])
        twist.twist.angular.z = float(av["z"])
        twist.twist.linear.x = float(lv["x"])
        twist.twist.linear.y = float(lv["y"])
        twist.twist.linear.z = float(lv["z"])
        self.pub_twist.publish(twist)

        # ----------------
        # Pose
        # ----------------
        pose = PoseStamped()
        pose.header = now_header(self, "vision")
        p = obs["pose"]["position"]
        q = obs["pose"]["rotation"]
        pose.pose.position.x = float(p["x"])
        pose.pose.position.y = float(p["y"])
        pose.pose.position.z = float(p["z"])
        pose.pose.orientation.w = float(q["w"])
        pose.pose.orientation.x = float(q["x"])
        pose.pose.orientation.y = float(q["y"])
        pose.pose.orientation.z = float(q["z"])
        self.pub_pose.publish(pose)

        # ----------------
        # Cameras
        # ----------------
        cams = obs.get("cameras", {})        
        for cam_name, cam_data in cams.items():
            stamp = self.get_clock().now().to_msg()
            # Color
            if "color" in cam_data and cam_data["color"] is not None:
                img_np = cam_data["color"]  # HxWx3 uint8
                if isinstance(img_np, np.ndarray):
                    msg = np_to_ros_image(img_np, f"{cam_name}_color", stamp, encoding="bgr8")
                    msg.header = now_header(self, f"{cam_name}_color")
                    topic = f"/spot/cameras/{cam_name}/color"
                    self._get_img_pub(topic).publish(msg)

            # Depth
            if "depth" in cam_data and cam_data["depth"] is not None:
                depth_np = cam_data["depth"]
                if isinstance(depth_np, np.ndarray):
                    # depth
                    msg = np_to_ros_image(depth_np, f"{cam_name}_depth", stamp, encoding="16UC1")
                    msg.header = now_header(self, f"{cam_name}_depth")
                    topic = f"/spot/cameras/{cam_name}/depth"
                    self._get_img_pub(topic).publish(msg)


def main():
    rclpy.init()

    # ---- create your SpotAgent here ----
    # from your_module import SpotAgent
    agent = SpotAgent(hostname="137.146.188.170")

    node = SpotObservationPublisher(agent, publish_rate_hz=5.0)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()