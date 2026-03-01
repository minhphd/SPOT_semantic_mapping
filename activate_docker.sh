xhost +local:
docker run --rm -it \
  --gpus all \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e ROS_DOMAIN_ID=0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD":/app \
  ros_semantic_mapping:foxglove \
  bash

source /opt/ros/humble/setup.bash