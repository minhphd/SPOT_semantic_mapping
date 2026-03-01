# ROS 2 base (Ubuntu 22.04)
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# --- OS deps (build tools + common ROS image deps) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

# --- Install Miniconda ---
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL -o /tmp/miniconda.sh \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# --- Copy env specs early for Docker cache ---
COPY environment.yml .
COPY requirements.txt .

RUN conda tos accept --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --channel https://repo.anaconda.com/pkgs/r

# --- Create conda env ---
RUN conda env create -n ros_semantic_mapping -f environment.yml && \
    conda clean -a -y

# Use conda env by default (no activate needed)
ENV PATH=/opt/conda/envs/ros_semantic_mapping/bin:$PATH

# If you still need pip-only deps
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U "jax[cuda12]"
RUN pip install accelerate

# Copy your code
COPY . .

# --- Make sure ROS is sourced in every shell ---
# This ensures ros2 commands work without manual sourcing
RUN printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -e' \
  'source /opt/ros/humble/setup.bash' \
  'exec "$@"' \
  > /ros_entrypoint.sh && chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
