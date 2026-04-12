import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image, BatteryState
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Image as RosImage

# Import your new Fast agent
from agent import FastSpotGraphAgent
from configs.loader import cfg

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

class SpotObservationPublisher(Node):
    def __init__(self, spot_agent, publish_rate_hz=5.0):
        super().__init__("spot_observation_publisher")
        self.agent = spot_agent

        # --- Pre-allocate static publishers ---
        self.pub_pose = self.create_publisher(PoseStamped, "/spot/pose", 10)
        self.pub_graph = self.create_publisher(String, "/spot/graph", 10)
        self.pub_twist = self.create_publisher(TwistStamped, "/spot/twist", 10)
        self.pub_batt = self.create_publisher(BatteryState, "/spot/battery", 10)
        
        # Pre-allocate image publishers for known Spot cameras
        self.pub_img = {}
        # We know Spot's standard cameras from the agent class
        for cam_name in self.agent.cameras.keys():
            self.pub_img[f"{cam_name}_color"] = self.create_publisher(Image, f"/spot/cameras/{cam_name}/color", 10)
            self.pub_img[f"{cam_name}_depth"] = self.create_publisher(Image, f"/spot/cameras/{cam_name}/depth", 10)

        # Start timer tick (No more asyncio loops needed!)
        self.period = 1.0 / float(publish_rate_hz)
        self.timer = self.create_timer(self.period, self._tick)

    def _tick(self):
        # 1) Instantly grab the thread-safe snapshot from the agent
        obs = self.agent.get_observation()

        # At startup, the lanes might still be empty for a fraction of a second
        if obs is None or not obs.get("cameras"):
            return

        # 2) Get ONE synchronized timestamp for this exact snapshot
        sync_stamp = self.get_clock().now().to_msg()

        def create_header(frame_id):
            h = Header()
            h.stamp = sync_stamp
            h.frame_id = frame_id
            return h

        # ----------------
        # Battery
        # ----------------
        if obs.get("battery") is not None:
            b = BatteryState()
            b.header = create_header("spot_body")
            b.percentage = float(obs["battery"]) / 100.0
            self.pub_batt.publish(b)

        # ----------------
        # Twist
        # ----------------
        if obs.get("imu") is not None and obs["imu"].get("angular_velocity"):
            twist = TwistStamped()
            twist.header = create_header("spot_body")
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
        if obs.get("pose") is not None and obs["pose"].get("position"):
            pose = PoseStamped()
            pose.header = create_header("vision")
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
        # Graph Text Description
        # ----------------
        if obs.get('g_text_desc'):
            msg = String()
            msg.data = obs['g_text_desc']
            self.pub_graph.publish(msg)
            
        # ----------------
        # Cameras
        # ----------------
        cams = obs.get("cameras", {})        
        for cam_name, cam_data in cams.items():
            # Color
            if "color" in cam_data and cam_data["color"] is not None:
                img_np = cam_data["color"]
                if isinstance(img_np, np.ndarray):
                    msg = np_to_ros_image(img_np, f"{cam_name}_color", sync_stamp, encoding="bgr8")
                    msg.header = create_header(f"{cam_name}_color")
                    
                    pub_key = f"{cam_name}_color"
                    if pub_key in self.pub_img:
                        self.pub_img[pub_key].publish(msg)

            # Depth
            if "depth" in cam_data and cam_data["depth"] is not None:
                depth_np = cam_data["depth"]
                if isinstance(depth_np, np.ndarray):
                    msg = np_to_ros_image(depth_np, f"{cam_name}_depth", sync_stamp, encoding="16UC1")
                    msg.header = create_header(f"{cam_name}_depth")
                    
                    pub_key = f"{cam_name}_depth"
                    if pub_key in self.pub_img:
                        self.pub_img[pub_key].publish(msg)


def main():
    rclpy.init()

    # Use the FastSpotGraphAgent instead of the old one
    # delays=(fast, med, slow) -> 0.05s (20Hz) for images, 1.0s (1Hz) for battery, 2.0s (0.5Hz) for Graph
    agent = FastSpotGraphAgent(cfg, hostname="137.146.188.170", delays=(0.05, 1.0, 2.0))

    # The ROS node can now publish comfortably at 10Hz or higher without lagging
    node = SpotObservationPublisher(agent, publish_rate_hz=50.0)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()