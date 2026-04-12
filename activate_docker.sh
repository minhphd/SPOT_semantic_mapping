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
pip install -U jax[cuda12] accelerate
python3 -m pip install bosdyn-client bosdyn-mission bosdyn-choreography-client
pip install -e .