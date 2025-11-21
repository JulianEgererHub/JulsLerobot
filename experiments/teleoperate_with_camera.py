from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.teleoperators.so101_leader import SO101LeaderConfig,SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
import rerun as rr


camera_config = {
    "front": OpenCVCameraConfig(
    index_or_path="/dev/video0",
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
),
    "context": RealSenseCameraConfig(
    serial_number_or_name="405622073533",
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    use_depth=True,
    rotation=Cv2Rotation.NO_ROTATION
)
}

teleop_config = SO101LeaderConfig(
    port='/dev/ttyACM1',
    id="leader1",
)

robot_config = SO101FollowerConfig(
    port='/dev/ttyACM0',
    id="follower1",
    cameras=camera_config
)

# camera = RealSenseCamera(config)
# camera.connect()

# # Capture a color frame via `read()` and a depth map via `read_depth()`.
# try:
#     color_frame = camera.read()
#     depth_map = camera.read_depth()

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

init_rerun(session_name="MySession")

while True:
    observation = robot.get_observation()
    action = teleop_device.get_action(mirror_shoulder_pan=True)

    log_rerun_data(observation, action)
    robot.send_action(action)

